#include <iostream>
#include <iomanip>
#include <ctime>

using namespace std;

const int qSize = 6; // number of states(rooms in this problem)
const double gamma = 0.8; // the discount factor 
const int iterations = 10;
int initialStates[qSize] = {1, 3, 5, 2, 4, 0};

// rewards of every (state, action) pair
// and out destination is door 5. 
int R[qSize][qSize] = {{-1, -1, -1, -1, 0, -1},  
                       {-1, -1, -1, 0, -1, 100}, 
                       {-1, -1, -1, 0, -1, -1}, 
                       {-1, 0, 0, -1, 0, -1}, 
                       {0, -1, -1, 0, -1, 100}, 
                       {-1, 0, -1, -1, 0, 100}};

int Q[qSize][qSize];
int currentState;

void episode(int initialState);
void chooseAnAction();
int getRandomAction(int upperBound, int lowerBound);
void initialize();
int maximum(int state, bool returnIndexOnly);
int reward(int action);

int main()
{
	int newState;
	initialize();

	// Perform learning trials starting 
	// at all initial states
	for(int j = 0; j <= (iterations - 1); j++)
		for(int i = 0; i <= (qSize - 1); i++)
		{
			episode(initialStates[i]);
		}

	// Print out Q matrix
	cout << "Q matrix values:" << endl;
	for(int i = 0; i <= (qSize - 1); i++)
	{
		for(int j = 0; j <= (qSize - 1); j++)
		{
			cout << setw(5) << Q[i][j];
			if(j < qSize - 1)
				cout << ",";
		}
		cout << "\n";
	}

	// Perform tests, starting at all initial states
	cout <<endl << "Shortest routes from initial states:" << endl;
	for(int i = 0; i <= (qSize - 1); i++)
	{
		currentState = initialStates[i];
		newState = 0;
		do
		{
			newState = maximum(currentState, true);
			cout << currentState << ", ";
			currentState = newState;
		}while(currentState != 5);
		cout << "5" << endl;
	}

	return 0;
}

void episode(int initialState)
{
	// travel from state to state until 
	// goal state is reached 
	currentState = initialState;
	do{
		chooseAnAction();
	}while(currentState != 5);

	// When currentState = 5, run through 
	// the set once more for convergence
	for(int i = 0; i <= (qSize - 1); i++)
		chooseAnAction();
}

void chooseAnAction()
{
	int possibleAction;
	possibleAction = getRandomAction(qSize, 0);

	if(R[currentState][possibleAction] >= 0)
	{
		// set newest Q-value 
		Q[currentState][possibleAction] = reward(possibleAction);
		currentState = possibleAction;
	}
}

// Randomly choose a possible action 
// connected to the current state
int getRandomAction(int upperBound, int lowerBound)
{
	int action;
	bool choiceIsValid = false;
	int range = (upperBound - lowerBound);

	do{
		// Get a random value between 0 and 6
		action = lowerBound + int(range * (rand() / (RAND_MAX + 1.0)));
		action = lowerBound + int(range * (rand() / (RAND_MAX + 1.0)));
		if(R[currentState][action] > -1)
		{
			choiceIsValid = true;
		}
	}while(choiceIsValid == false);
	return action;
}

void initialize()
{
	srand((unsigned)time(0));  // reset the seed for rand()
	for(int i = 0; i <= (qSize - 1); i ++)
		for(int j = 0; j <= (qSize - 1); j ++)
			Q[i][j] = 0;
}

// This function gets the maximum Q-value 
// for a state, which equals to get the max  
// value for a cetain row in matrix Q 
int maximum(int state, bool returnIndexOnly)
{
	// if returnIndexOnly = true, a Q matrix index is returned
	// if returnIndexOnly = false, a Q matrix element is returned

	int winner;
	bool foundNewWinner;
	bool done = false;

	winner = 0;
	do{
		foundNewWinner = false;
		for(int i = 0; i <= (qSize - 1); i++)
			if((i < winner) || (i > winner)) // avoid self comparison
				if(Q[state][i] > Q[state][winner])
				{
					winner = i;
					foundNewWinner = true;
				}

		if(foundNewWinner == false)
			done = true;
	}while(done == false);

	if(returnIndexOnly == true)
		return winner;
	else
		return Q[state][winner];
}

// get the new Q-value for the current 
// state taking a certain action 
int reward(int action)
{
	return static_cast<int>(R[currentState][action] + (gamma * maximum(action, false)));
}


