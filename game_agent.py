"""
"""
import random
import math


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


'''
    ______________
    Preliminaries
    ______________

        * The isolation game is fully deterministic. Given hypothetical infinite resources,
          the game state can be determined a priori by examining and scoring all available options.

        * According to problem description, a move can move in L shape if the target square is
          valid. This indeed simplifies forcasting legitimate moves.

          Therefore, the following rules apply:

              0. Given the L-shape restriction, the next valid move, cannot be found on either
                 diagonal of current peice position.

              1. If a player at any of the (4) corners, then he would only have 2 valid moves

              2. If a player at the center of at least (5x5) board, then he would have the maximum
                 number of moves (8)

              3. If a player is at one of the tiles (border lines), then at Max he would have (4)
                 valid moves.

        * One maybe tempted to assign a low score to a player if he is at a corner, and a high one
          if he was at the center. However, an excellent hueristic should take into account the
          probability of landing on a next better square given the current position. For example,
          a player should be given a high score if he was at a corner, and his next move would be
          to the center. On contrary, a player should be given a low score if he was at the center
          and his next move would be toward the corner!.

        * Let N be the leading dimension of the board of size NxN. The following results can be
          easily shown (I already did the analysis on papers and I am only showing the results here)

          Given a free NxN game of with N at least 5. Then

              1. The total number of (8) possible moves is (N-4)**2
              2. The total number of (6) possible moves is (N-3) * 4
              3. The total number of (4) possible moves is (N-2) * 4
              4. The total number of (2) possible moves is at maximum 8.

            For example, and without loss of generality, consider a 7x7 Grid,

              1. The total number of (8) possible moves is (7-4)**2 = 9
              2. The total number of (6) possible moves is (7-3) *4 = 16
              3. The total number of (4) possible moves is (7-2) *4 = 20

        * Also, a good hueristic should keep track of the opponent moves, so that we limit his options as he play.
'''

def get_probabilities(game):

    '''
        * Please Review the Preliminaries Section for more information and various assumptions
          the following function retuns the probability of available moves given the current postion

        * The available moves given their current postion should be adjusted as the game advances. That is to say,
          the number of valid 8 position square for example should be lowered as more spaces are being occupied.
          In other words, as the move advances, the original board is shrunk in size
    '''


    estimated_board_length = math.sqrt(len(game.get_blank_spaces()))

    possible_8_moves = (estimated_board_length-4)**2
    possible_6_moves = (estimated_board_length-3)*4
    possible_4_moves = (estimated_board_length-2)*4
    possible_2_moves = 8


    if(math.floor(estimated_board_length)==4):
        possible_8_moves = 0

    if(math.floor(estimated_board_length)==3):
        possible_6_moves =0

    if(math.floor(estimated_board_length)==2):
        possible_4_moves = 0

    possible_moves_marginal = possible_8_moves+possible_6_moves+possible_4_moves+possible_2_moves

    prob_8 = possible_8_moves/possible_moves_marginal
    prob_6 = possible_6_moves/possible_moves_marginal
    prob_4 = possible_4_moves/possible_moves_marginal
    prob_2 = possible_2_moves/possible_moves_marginal

    #assuming all events are independent

    prob_dict = {'8':prob_8, '6':prob_6, '4':prob_4, '2':prob_2}

    return prob_dict

#-----------------------------------------------------------------------------
#____________________________________________________________________________

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """


    '''
       Combines the lessons learnt from the previous two heuristics by merging both probabilistic forecasting
       and spatial information.
            1.    The best move at the beginning at game would at the center. It gives the maximum number
                  of options for the next move. If the board is 5x5 or more, then MAX player will have at
                  least 7 more moves for his next turn. The player who controls the center of the game,
                  is more likely to win. Therefore, any move towards the center should be highly rewarded.

            2.    The final score is formed by combining various portions of the previous scores and
                  this new established weight.

    '''


    # Have we won the game?
    if game.is_winner(player):
        return float("inf")

    # Do we even have moves to play?
    if game.is_loser(player):
        return float("-inf")


    #----------------Score I: How Far From The Center---------------

    game_center = (game.height//2, game.width//2)

    my_location = game.get_player_location(player)
    me_off_center= math.hypot(game_center[0] - my_location[0], game_center[1] - my_location[1])

    opp_location = game.get_player_location(game.get_opponent(player))
    opp_off_center = math.hypot(game_center[0] - opp_location[0], game_center[1] - opp_location[1])

    Bord_max_dim = max(game.height//2, game.width//2)

    # Add more weight if the opponent is near the center. The opp. score is to be subtracted in the end
    my_score_1  = Bord_max_dim - me_off_center
    opp_score_1 = Bord_max_dim - opp_off_center + 0.1*Bord_max_dim


    #----------------Score II Spatial Information ---------------------------

    score_2 = custom_score_2(game, player)

    #----------------Score III Next Move Probability----------------------------------

    score_3 = custom_score_3(game, player)
    #--------------------------------------------- ---------------------------

    alpha = 1.75
    beta =  1.35
    gamma = 1.4

    final_score = alpha*(my_score_1 - opp_score_1) + beta*score_2 + gamma*score_3

    return final_score


#--------------------------------------------- ---------------------------

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    """
        The following heuristic considers solely spatial information. The following applies.
            1.	Detect the location of the opponent, and assign a high score if my next position
                is in his next valid moves list. By this we aim at limiting his options and forcing
                him to go in a certain direction.
            2.	If my opponent is at either an edge or a corner and many squares are occupied already,
                then this is a good sign, the returned score should be high
            3.	If on the other hand, I was at an edge or a corner towards the end of the game,
                I should penalize my move severely in order not to be trapped with limited options.

    """


    # Have we won the game?
    if game.is_winner(player):
        return float("inf")

    # Do we even have moves to play?
    if game.is_loser(player):
        return float("-inf")


    opp_moves = game.get_legal_moves(game.get_opponent(player))
    my_moves = game.get_legal_moves(player)

    counter = 0
    for move in my_moves:
        if move in opp_moves:
            counter = counter+1

    my_score_1 = counter*5
    my_score_2 = 0

    game_progress = len(game.get_blank_spaces())/(game.height* game.width)

    #roughly predict if a given player is on an edge or a corner
    num_my_moves = len(my_moves)
    num_opp_moves = len(opp_moves)

    if(game_progress> 0.75):
        if(num_opp_moves == 4):
            my_score_2 = 3
        elif(num_opp_moves == 3):
            my_score_2 = 5
        elif(num_opp_moves <=2):
            my_score_2 = 7


        if(num_my_moves == 4):
            my_score_2 = -5
        elif(num_my_moves == 3):
            my_score_2 = -10
        elif(num_my_moves <=2):
            my_score_2 = -15


    #print("score1", my_score_1,"score2", my_score_2 )
    #combining with improved score
    final_score = float((len(my_moves) - len(opp_moves)) + (my_score_1 + my_score_2))
    
    return final_score


#--------------------------------------------- ---------------------------


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """


    '''
        The following heuristic relates the probability of available moves given their current position to both
        the player and the opponent next moves. The number of next valid moves for both players is weighted by the
        returned probability. If the opponent score is greater than student score, the final return score is downgraded.
        On the other hand, if student score is higher, the returned score is emphasized by multiplying it by another weight.
        The cutting threshold and both weights are experimentally determined and tuned.
    '''
    # Have we won the game?
    if game.is_winner(player):
        return float("inf")

    # Do we even have moves to play?
    if game.is_loser(player):
        return float("-inf")


    Bord_min_dim = min(game.height//2, game.width//2)

    prob_dict = get_probabilities(game)
    prob_score = 8*prob_dict['8']+ 6*prob_dict['6'] - 4*prob_dict['4'] - 2*prob_dict['2']

    my_moves = len(game.get_legal_moves(player))
    my_predictions = my_moves*prob_score

    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    opp_predictions = opp_moves*prob_score

    #this does not give good prediction if legal moves are both equal
    my_score = 0
    opp_score = 0

    thrsh = 0
    if(opp_moves != my_moves):

        if (my_predictions- opp_predictions) >= thrsh: # more chances of having better next moves
            my_score = abs(my_predictions) * Bord_min_dim
        else:
            opp_score =  abs(opp_predictions) * Bord_min_dim *0.5


    return (my_score - opp_score)
#_____________________________________________________________________________

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move


    #__________________________________________________________________________
    def my_minimax(self, game, depth, maximizing_player=True):

        '''
        References:
        ------------
            * https://en.wikipedia.org/wiki/Minimax
            * https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
            * https://github.com/aimacode/aima-python/blob/master/games.py
        '''
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()


        # get a list of the legal moves available to the active player
        next_move = (-1, -1)
        possible_moves = game.get_legal_moves()


        if not possible_moves:
            '''
                The game has a utility of +inf if the player has won,
                a value of -inf if the player has lost, and a value of 0
                otherwise.
            '''
            return game.utility(self), None

        if depth == 0:
            return self.score(game, self), next_move


        #searching all possible children using recursion
        if maximizing_player:

            max_value = -math.inf

            for move in possible_moves:

                # get a successor of the current state by making a copy of the board
                new_board = game.forecast_move(move)
                v, _ = self.my_minimax(new_board, depth-1, maximizing_player = False)

                if v >= max_value:
                    max_value, next_move = v, move

            return max_value, next_move

        else:

            min_value = math.inf

            for move in possible_moves:

                new_board = game.forecast_move(move)
                v, _ = self.my_minimax(new_board, depth-1, maximizing_player=True)

                if v < min_value:
                    min_value, next_move = v, move

            return min_value, next_move

    #__________________________________________________________________________

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()


        possible_moves = game.get_legal_moves()

        if not possible_moves:
            return (-1,-1)
        else:
            _, best_move = self.my_minimax(game, depth, maximizing_player=True)
            return best_move


#__________________________________________________________________________


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        next_move = (-1, -1)

        possible_moves = game.get_legal_moves()
        if not possible_moves:
            return (-1, -1)

        # Clearly the best move at the beginning at game would at the center. It gives the maximum number
        # of options for the next move. If the board is 5x5 or more, then MAX player will have at least 7 more moves
        # for his next turn.

        if game.move_count == 0:
            return(game.height//2, game.width//2)



        try:

            current_depth = 1

            while True:

                next_move =  self.alphabeta(game, current_depth)
                current_depth += 1

        except SearchTimeout:
            pass

        # Return the best move from the last completed search iteration
        return next_move

    #________________________________________________________________________________

    def my_alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):

        '''
        References:
            * https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
            * https://github.com/aimacode/aima-pseudocode/blob/master/md/Iterative-Deepening-Search.md
       
        '''

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()


        next_move = (-1, -1)
        possible_moves = game.get_legal_moves()

        if not possible_moves:
            '''
                The game has a utility of +inf if the player has won,
                a value of -inf if the player has lost, and a value of 0
                otherwise.
            '''
            return game.utility(self), None

        if depth == 0:
            return self.score(game, self), next_move



        #searching all possible children using recursion
        if maximizing_player:

            max_value = -math.inf

            for move in possible_moves:

                new_board = game.forecast_move(move)
                v, _ = self.my_alphabeta(new_board, depth-1, alpha, beta, maximizing_player = False)

                if v >= max_value:
                    max_value, next_move = v, move

                #update alpha with the highest value of v
                alpha = max(alpha,max_value)

                if(beta <= alpha):
                    break

            return max_value, next_move

        else:

            min_value = math.inf

            for move in possible_moves:

                new_board = game.forecast_move(move)
                v, _ = self.my_alphabeta(new_board, depth-1, alpha, beta, maximizing_player=True)

                if v < min_value:
                    min_value, next_move = v, move

                beta = min(beta, min_value)

                if beta <= alpha:
                    break

            return min_value, next_move

    #__________________________________________________________________________________________________

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.

        References:
        -----------
            * https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
            * https://github.com/aimacode/aima-pseudocode/blob/master/md/Iterative-Deepening-Search.md
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        possible_moves = game.get_legal_moves()

        if not possible_moves:
            return (-1,-1)
        else:
            _, best_move = self.my_alphabeta(game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True)
            return best_move
