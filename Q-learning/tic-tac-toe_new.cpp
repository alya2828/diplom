#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cmath>
using namespace std;

const int EMPTY = -1;
const int X = 1;
const int O = 0;
const int CELLS = 9;

//  3^9 состояний
const int STATES = 19683;

// Параметры обучения.
const double ALPHA = 0.1;    // Скорость обучения
const double GAMMA = 0.95;   // Важность будущих наград
const double EPSILON = 0.1;  // Вероятность случайного хода при обучении

const int TRAINING_GAMES = 100000;
const int TEST_GAMES = 10000;
const int MINIMAX_TEST_GAMES = 10000;
const unsigned int RANDOM_SEED = 42;

// Для независимых экспериментов оставляем false.
// true — продолжить обучение на ранее сохранённой таблице того же соперника
const bool CONTINUE_TRAINING = false;

// Файлы для разных способов обучения
const char* TABLE_FILES[3] = {
    "table_qlearning_random.txt",
    "table_qlearning_minimax.txt",
    "table_qlearning_mixed.txt"
};
const char* RESULTS_FILES[3] = {
    "results_random.csv",
    "results_minimax.csv",
    "results_mixed.csv"
};
const char* OLD_TABLE_FILE = "table_qlearning.txt";

double Q[STATES][CELLS];
bool visited[STATES];

// Сохраняем уже вычисленные оценки Minimax
int minimaxCache[STATES][2];
bool minimaxReady[STATES][2];

// Значения клеток сдвигаются на 1, чтобы получить цифры троичной записи.
int stateNumber(const int state[CELLS])
{
    int number = 0;
    for (int i = 0; i < CELLS; i++)
        number = number * 3 + (state[i] + 1);
    return number;
}

// Восстановить девять клеток по номеру состояния для сохранения таблицы
void numberToState(int number, int state[CELLS])
{
    for (int i = CELLS - 1; i >= 0; i--)
    {
        state[i] = (number % 3) - 1;
        number = number / 3;
    }
}

// Проверить победу указанного игрока 
int checkWin(const int state[CELLS], int player)
{
    if (state[0] == player && state[1] == player && state[2] == player) return 1;
    if (state[3] == player && state[4] == player && state[5] == player) return 1;
    if (state[6] == player && state[7] == player && state[8] == player) return 1;

    if (state[0] == player && state[3] == player && state[6] == player) return 1;
    if (state[1] == player && state[4] == player && state[7] == player) return 1;
    if (state[2] == player && state[5] == player && state[8] == player) return 1;

    if (state[0] == player && state[4] == player && state[8] == player) return 1;
    if (state[2] == player && state[4] == player && state[6] == player) return 1;
    return 0;
}

// Проверить, заполнено ли поле 
int isFull(const int state[CELLS])
{
    for (int i = 0; i < CELLS; i++)
        if (state[i] == EMPTY) return 0;
    return 1;
}

// Случайный допустимый ход выбираем только среди свободных клеток.
int getRandomMove(const int state[CELLS])
{
    int freeCells[CELLS];
    int count = 0;

    for (int i = 0; i < CELLS; i++)
    {
        if (state[i] == EMPTY)
        {
            freeCells[count] = i;
            count++;
        }
    }

    if (count == 0) return -1;
    return freeCells[rand() % count];
}

int minimax(int state[CELLS], bool turnO)
{
    if (checkWin(state, O)) return 1;
    if (checkWin(state, X)) return -1;
    if (isFull(state)) return 0;

    int number = stateNumber(state);
    int turn = turnO ? 1 : 0;
    if (minimaxReady[number][turn]) return minimaxCache[number][turn];

    int best = turnO ? -2 : 2;

    for (int i = 0; i < CELLS; i++)
    {
        if (state[i] != EMPTY) continue;

        state[i] = turnO ? O : X;
        int score = minimax(state, !turnO);
        state[i] = EMPTY; 

        if (turnO && score > best) best = score;
        if (!turnO && score < best) best = score;
    }
    minimaxReady[number][turn] = true;
    minimaxCache[number][turn] = best;
    return best;
}

// Выбрать ход O, который гарантирует лучший доступный исход
// Если равноценных ходов несколько, один выбирается случайно
int getMinimaxMove(int state[CELLS])
{
    int bestScore = -2;
    int bestMoves[CELLS];
    int count = 0;

    for (int i = 0; i < CELLS; i++)
    {
        if (state[i] != EMPTY) continue;

        state[i] = O;
        int score = minimax(state, false);
        state[i] = EMPTY;

        if (score > bestScore)
        {
            bestScore = score;
            count = 0;
            bestMoves[count++] = i;
        }
        else if (score == bestScore)
        {
            bestMoves[count++] = i;
        }
    }

    if (count == 0) return -1;
    return bestMoves[rand() % count];
}

// Выбрать свободную клетку с наибольшим Q-значением
int getBestMove(const int state[CELLS])
{
    int number = stateNumber(state);
    int bestMove = -1;

    for (int i = 0; i < CELLS; i++)
    {
        if (state[i] == EMPTY)
        {
            if (bestMove == -1 || Q[number][i] > Q[number][bestMove])
                bestMove = i;
        }
    }
    return bestMove;
}

// Максимальная оценка доступного хода в данном состоянии
double getMaxQ(const int state[CELLS])
{
    int bestMove = getBestMove(state);
    if (bestMove == -1) return 0.0;
    return Q[stateNumber(state)][bestMove];
}

int getAction(const int state[CELLS], bool training)
{
    double probability = (double)rand() / RAND_MAX;
    if (training && probability < EPSILON)
        return getRandomMove(state);
    return getBestMove(state);
}

// Обновить Q-значение по формуле Q-learning (с учётом будущей награды)
void updateQ(int oldNumber, int action, double reward,
             bool gameOver, const int nextState[CELLS])
{
    double target = reward;
    if (!gameOver)
        target += GAMMA * getMaxQ(nextState);

    Q[oldNumber][action] += ALPHA * (target - Q[oldNumber][action]);
}

// Провести партию. opponent: 1 — Random, 2 — Minimax, 3 — смешанный.
int playGame(bool training, int opponent)
{
    int state[CELLS];
    for (int i = 0; i < CELLS; i++) state[i] = EMPTY;

    if (opponent == 3)
        opponent = (rand() % 2 == 0) ? 1 : 2;

    while (true)
    {
        int oldNumber = stateNumber(state);
        if (training) visited[oldNumber] = true;

        int action = getAction(state, training);
        state[action] = X;

        if (checkWin(state, X))
        {
            if (training) updateQ(oldNumber, action, 1.0, true, state);
            return 1;
        }
        if (isFull(state))
        {
            if (training) updateQ(oldNumber, action, 0.0, true, state);
            return 0;
        }

        // Во время обучения и тестирования используется выбранный соперник
        int botMove = (opponent == 2) ? getMinimaxMove(state) : getRandomMove(state);
        state[botMove] = O;

        if (checkWin(state, O))
        {
            if (training) updateQ(oldNumber, action, -1.0, true, state);
            return -1;
        }
        if (isFull(state))
        {
            if (training) updateQ(oldNumber, action, 0.0, true, state);
            return 0;
        }

        // Игра продолжается учитываем оценку следующего состояния
        if (training) updateQ(oldNumber, action, 0.0, false, state);
    }
}

// Подсчитать количество посещённых состояний
int countStates()
{
    int count = 0;
    for (int i = 0; i < STATES; i++)
        if (visited[i]) count++;
    return count;
}

// Протестировать агента без изменения Q-таблицы
void testModel(int results[3], bool strongBot, int games)
{
    results[0] = 0; // Победы
    results[1] = 0; // Ничьи
    results[2] = 0; // Поражения

    for (int i = 0; i < games; i++)
    {
        int outcome = playGame(false, strongBot ? 2 : 1);
        if (outcome == 1) results[0]++;
        else if (outcome == 0) results[1]++;
        else results[2]++;
    }
}

bool saveTable(const char* filename)
{
    ofstream file(filename);
    if (!file.is_open()) return false;

    file << setprecision(12);
    int state[CELLS];

    for (int number = 0; number < STATES; number++)
    {
        if (!visited[number]) continue;

        numberToState(number, state);
        for (int i = 0; i < CELLS; i++) file << state[i] << " ";
        for (int i = 0; i < CELLS; i++)
        {
            file << Q[number][i];
            if (i < CELLS - 1) file << " ";
        }
        file << "\n";
    }
    return file.good();
}

// Загрузить ранее сохранённую таблицу для продолжения обучения
bool loadTable(const char* filename)
{
    ifstream file(filename);
    if (!file.is_open()) return false;

    int state[CELLS];
    double values[CELLS];
    int count = 0;

    while (file >> state[0])
    {
        for (int i = 1; i < CELLS; i++)
            if (!(file >> state[i])) return false;

        for (int i = 0; i < CELLS; i++)
            if (!(file >> values[i])) return false;

        for (int i = 0; i < CELLS; i++)
        {
            if (state[i] < EMPTY || state[i] > X) return false;
            if (!isfinite(values[i]) || values[i] < -1.000001 || values[i] > 1.000001)
                return false;
        }

        int number = stateNumber(state);
        visited[number] = true;
        for (int i = 0; i < CELLS; i++) Q[number][i] = values[i];
        count++;
    }

    if (!file.eof()) return false;
    cout << "Loaded " << count << " states from " << filename << "\n";
    return true;
}

bool saveComparison(const int randomResults[3], const int minimaxResults[3], const char* filename)
{
    ofstream file(filename);
    if (!file.is_open()) return false;

    file << "opponent,test_games,wins,draws,losses\n";
    file << "random," << TEST_GAMES << "," << randomResults[0] << ","
         << randomResults[1] << "," << randomResults[2] << "\n";
    file << "minimax," << MINIMAX_TEST_GAMES << "," << minimaxResults[0] << ","
         << minimaxResults[1] << "," << minimaxResults[2] << "\n";
    return file.good();
}

// Записать результаты теста с выбранным соперником
bool saveTestResults(const char* opponent, const int results[3], int games, const char* filename)
{
    ofstream file(filename);
    if (!file.is_open()) return false;

    file << "opponent,test_games,wins,draws,losses\n";
    file << opponent << "," << games << "," << results[0] << ","
         << results[1] << "," << results[2] << "\n";
    return file.good();
}

// Выполнить итоговую проверку против выбранного соперника
// Если результаты уже получены на последней контрольной точке, используем их повторно
void runSelectedTest(int opponent, const int* savedRandom,
                     const int* savedMinimax, const char* resultSuffix)
{
    int randomResults[3];
    int minimaxResults[3];
    string comparisonFile = string("comparison_") + resultSuffix + ".csv";
    string testFile = string("test_") + resultSuffix + ".csv";

    if (opponent == 1 || opponent == 3)
    {
        if (savedRandom != 0)
        {
            for (int i = 0; i < 3; i++) randomResults[i] = savedRandom[i];
        }
        else
        {
            testModel(randomResults, false, TEST_GAMES);
        }

        cout << "Random bot: wins=" << randomResults[0]
             << ", draws=" << randomResults[1]
             << ", losses=" << randomResults[2] << "\n";
    }

    if (opponent == 2 || opponent == 3)
    {
        if (savedMinimax != 0)
        {
            for (int i = 0; i < 3; i++) minimaxResults[i] = savedMinimax[i];
        }
        else
        {
            testModel(minimaxResults, true, MINIMAX_TEST_GAMES);
        }

        cout << "Minimax: wins=" << minimaxResults[0]
             << ", draws=" << minimaxResults[1]
             << ", losses=" << minimaxResults[2] << "\n";
    }

    bool saved;
    if (opponent == 3)
    {
        saved = saveComparison(randomResults, minimaxResults, comparisonFile.c_str());
        cout << (saved ? "Comparison saved to " : "Cannot save ")
             << comparisonFile << "\n";
    }
    else if (opponent == 1)
    {
        saved = saveTestResults("random", randomResults, TEST_GAMES, testFile.c_str());
        cout << (saved ? "Results saved to " : "Cannot save ")
             << testFile << "\n";
    }
    else
    {
        saved = saveTestResults("minimax", minimaxResults, MINIMAX_TEST_GAMES, testFile.c_str());
        cout << (saved ? "Results saved to " : "Cannot save ")
             << testFile << "\n";
    }
}

int main()
{
    srand(RANDOM_SEED);

    cout << "Choose mode:\n";
    cout << "1 - Train model and test it\n";
    cout << "2 - Test saved model (without training)\n";
    cout << "Your choice: ";

    int mode;
    if (!(cin >> mode) || (mode != 1 && mode != 2))
    {
        cout << "Invalid mode\n";
        return 1;
    }

    int trainingOpponent = 1;
    const char* tableFile = OLD_TABLE_FILE;
    const char* resultsFile = 0;
    const char* resultSuffix = "old";

    if (mode == 1)
    {
        cout << "\nChoose opponent for TRAINING:\n";
        cout << "1 - Random bot\n";
        cout << "2 - Minimax\n";
        cout << "3 - Mixed (50% Random, 50% Minimax)\n";
        cout << "Your choice: ";

        if (!(cin >> trainingOpponent) || trainingOpponent < 1 || trainingOpponent > 3)
        {
            cout << "Invalid training opponent\n";
            return 1;
        }

        tableFile = TABLE_FILES[trainingOpponent - 1];
        resultsFile = RESULTS_FILES[trainingOpponent - 1];
        if (trainingOpponent == 1) resultSuffix = "random";
        if (trainingOpponent == 2) resultSuffix = "minimax";
        if (trainingOpponent == 3) resultSuffix = "mixed";
    }
    else
    {
        cout << "\nChoose saved Q-table:\n";
        cout << "1 - Trained against Random bot\n";
        cout << "2 - Trained against Minimax\n";
        cout << "3 - Mixed training\n";
        cout << "4 - Previous table_qlearning.txt\n";
        cout << "Your choice: ";

        int savedModel;
        if (!(cin >> savedModel) || savedModel < 1 || savedModel > 4)
        {
            cout << "Invalid saved model\n";
            return 1;
        }
        if (savedModel < 4)
        {
            tableFile = TABLE_FILES[savedModel - 1];
            if (savedModel == 1) resultSuffix = "random";
            if (savedModel == 2) resultSuffix = "minimax";
            if (savedModel == 3) resultSuffix = "mixed";
        }
    }

    cout << "\nChoose opponent for TESTING:\n";
    cout << "1 - Random bot\n";
    cout << "2 - Minimax\n";
    cout << "3 - Both (comparison)\n";
    cout << "Your choice: ";

    int testOpponent;
    if (!(cin >> testOpponent) || testOpponent < 1 || testOpponent > 3)
    {
        cout << "Invalid test opponent\n";
        return 1;
    }

    if (mode == 2)
    {
        if (!loadTable(tableFile))
        {
            cout << "Cannot load " << tableFile << "\n";
            return 1;
        }
        runSelectedTest(testOpponent, 0, 0, resultSuffix);
        return 0;
    }

    if (CONTINUE_TRAINING && !loadTable(tableFile))
    {
        cout << "Cannot load a valid table from " << tableFile << "\n";
        return 1;
    }

    ofstream csv(resultsFile);
    if (!csv.is_open())
    {
        cout << "Cannot create " << resultsFile << "\n";
        return 1;
    }
   
    csv << "training_games,random_wins,random_draws,random_losses,"
           "minimax_wins,minimax_draws,minimax_losses,visited_states\n";

    const int checkpoints[4] = {0, 1000, 10000, TRAINING_GAMES};
    int completedGames = 0;
    int lastRandomResults[3] = {0, 0, 0};
    int lastMinimaxResults[3] = {0, 0, 0};

    // На каждой контрольной точке тестируем модель против Random и Minimax
    // Во время тестирования Q-таблица не изменяется
    for (int step = 0; step < 4; step++)
    {
        while (completedGames < checkpoints[step])
        {
            playGame(true, trainingOpponent);
            completedGames++;
        }

        int randomResults[3];
        int minimaxResults[3];
        testModel(randomResults, false, TEST_GAMES);
        testModel(minimaxResults, true, MINIMAX_TEST_GAMES);

        int states = countStates();
        cout << "After " << completedGames << " training games (states="
             << states << "):\n";
        cout << "  Random bot: wins=" << randomResults[0]
             << ", draws=" << randomResults[1]
             << ", losses=" << randomResults[2] << "\n";
        cout << "  Minimax:    wins=" << minimaxResults[0]
             << ", draws=" << minimaxResults[1]
             << ", losses=" << minimaxResults[2] << "\n";

        csv << completedGames << ","
            << randomResults[0] << "," << randomResults[1] << "," << randomResults[2] << ","
            << minimaxResults[0] << "," << minimaxResults[1] << "," << minimaxResults[2] << ","
            << states << "\n";

        // Сохраняем последние результаты, чтобы не проводить те же тесты второй раз.
        if (step == 3)
        {
            for (int i = 0; i < 3; i++)
            {
                lastRandomResults[i] = randomResults[i];
                lastMinimaxResults[i] = minimaxResults[i];
            }
        }
    }
    csv.close();

    if (!saveTable(tableFile))
    {
        cout << "Cannot save " << tableFile << "\n";
        return 1;
    }
    cout << "Table saved to " << tableFile << "\n";
    cout << "Learning results saved to " << resultsFile << "\n";

    cout << "\nFinal test:\n";
    runSelectedTest(testOpponent, lastRandomResults, lastMinimaxResults, resultSuffix);
    return 0;
}
