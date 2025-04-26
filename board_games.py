import numpy as np
import pickle
import time
import matplotlib.pylab as plt
import pdb
import board_games_fun as bfun
import board_graphical_interface as bgra


def board_game_test(game_object, strategy_x, strategy_o, number_of_games = 100, choose_random = []):

    num_win_x = 0
    num_win_o = 0
    num_draws = 0
    
    Games = []
    Rewards = []

    for game in range(number_of_games):                   # episodes loop
        #print("game = " + str(game))
        State = game_object.initial_state()               # initial state - empty board in tictac
        player = 1                                        # first movement by player 1(cross)
        if_end = False 
        step_number = 0
        States = []
        Actions = []

        States.append(State)

        while (if_end == False):                           # episode steps loop
            step_number += 1

            actions = game_object.actions(State, player)   

            if player == 1:
                strategy = strategy_x
            else:
                strategy = strategy_o
            
            action_nr , value = strategy.choose_action(State,player)
            if (action_nr == None) | (step_number in choose_random):
                action_nr = np.random.randint(len(actions))
      
            NextState, Reward =  game_object.next_state_and_reward(player, State, actions[action_nr])

            State = NextState                                        # move to next state
            Actions.append(actions[action_nr])
            States.append(State)                                     # board for game description

            player = 3 - player                                      # player changing

            if game_object.end_of_game(Reward, step_number,State,action_nr):      # win or draw
                if_end = True
                if Reward == 1:
                    num_win_x += 1
                elif Reward == -1:
                    num_win_o += 1
                elif Reward == 0:
                    num_draws += 1
                Rewards.append(Reward)
        Games.append([States, Actions])
    return num_win_x, num_win_o, num_draws, Games, Rewards


def experiment_par_train():
    print("\nUCZENIE DWOCH STRATEGII JEDNOCZESNIE\n")
    #game = bfun.Tictactoe()                      # game class object
    #game = bfun.Tictac_general(4,4,3,False)
    #game = bfun.Tictac_general(10,10,4,False)
    #game = bfun.Tictac_general(5,5,4,False)
    #game = bfun.Connect4()
    #game = bfun.Chess("szachy_plansza_3x3.txt")
    #game = bfun.Chess("szachy_plansza_4x4.txt")
    #game = bfun.Chess("szachy_plansza_5x5.txt")
    #game = bfun.Chess("szachy_plansza_5x3_bez_kroli.txt")
    #game = bfun.Chess("szachy_plansza_5x10.txt")
    #game = bfun.Chess("szachy_plansza_standardowa.txt")
    #game = bfun.Chess("szachy_plansza_14x14.txt")
    game = bfun.Chess("szachy_plansza_top.txt")

    strategy_x, strategy_o = board_game_train_Q2(game,players_to_train = [1,2], number_of_games = 10)

    bgra.play_with_strategy(game_object = game, strategy = strategy_o, str_player=2)

    print("test stategii uczonych jednocześniewl,:")
    num_win_x, num_win_o, num_draws, Games, Rewards = board_game_test(game,strategy_x,strategy_o,choose_random=[])
    print("liczby wygranych: x = "+str(num_win_x)+", o = "+str(num_win_o) + ", l.remisów = "+str(num_draws))
    game.print_test_to_file("gry_wyuczonych_strategii.txt",num_win_x, num_win_o, num_draws, Games, Rewards)
    
    print("test stategii x na częściowo losowej o:")
    t = []
    nwin_x = []
    nwin_o = []
    ndraws = []
    for i in range(10):
        epsilon = i/10
        t.append(epsilon)     

        strategy_o.make_epsilon_greedy(player=2,epsilon=epsilon)
        num_win_x, num_win_o, num_draws, Games, Rewards =\
              board_game_test(game, strategy_x, strategy_o)
        game.print_test_to_file("gry_x_vs_losowe_o_epsilon"+str(epsilon)+".txt",\
                                num_win_x, num_win_o, num_draws, Games, Rewards)
        
        nwin_x.append(num_win_x)
        nwin_o.append(num_win_o)
        ndraws.append(num_draws)
    strategy_o.make_pure()
    plt.plot(t,nwin_x,"x",t,nwin_o,"o",t, ndraws,"-")
    plt.title("tictac test results with Nash x strategy and partially random o strategy")
    plt.xlabel("randomness of o strategy (1 - full random)")
    plt.ylabel("number of games")
    plt.legend(["num.of x wins","num.of o wins","num of draws"])
    plt.savefig("test_Nash_x_strategy_random_o_strategy.png")
    fig1 = plt
    plt.show()

    print("test stategii o na częściowo losowej x:")
    t = []
    nwin_x = []
    nwin_o = []
    ndraws = []
    for i in range(10):
        epsilon = i/10
        t.append(epsilon)       
        strategy_x.make_epsilon_greedy(player=1,epsilon=epsilon)
        num_win_x, num_win_o, num_draws, Games, Rewards = \
            board_game_test(game, strategy_x,strategy_o)
        game.print_test_to_file("gry_losowe_x_epsilon"+str(epsilon)+"_vs_o.txt",\
                                num_win_x, num_win_o, num_draws, Games, Rewards)
        nwin_x.append(num_win_x)
        nwin_o.append(num_win_o)
        ndraws.append(num_draws)
    strategy_x.make_pure()
    plt.plot(t,nwin_x,"x",t,nwin_o,"o",t, ndraws,"-")
    plt.title("tictac test results with Nash o strategy and partially random x strategy")
    plt.xlabel("randomness of x strategy (1 - full random)")
    plt.ylabel("number of games")
    plt.legend(["num.of x wins","num.of o wins","num of draws"])
    plt.savefig("test_Nash_o_strategy_random_x_strategy.png")
    plt.show()
    fig2 = plt

    # play with x (white in chess) strategy:
    #bgra.play_with_strategy(game_object = game, strategy = strategy_x, str_player=1)

    # play with o (black in chess) strategy:
    bgra.play_with_strategy(game_object = game, strategy = strategy_o, str_player=2)

    # print("\nDouczanie strategii o na ustalonej strategii x, by sprawdzić")
    # print("na ile uczenie równoczesne było skuteczne.\n")
    # _, strategy_o_doucz = \
    #     board_game_train_Q2(game,players_to_train = [2], strategy_x = strategy_x, number_of_games = 10000)
    # strategy_o_doucz.to_file("strategy_o_doucz.txt",)
    # print("test stategii douczanej o i uczonej x:")
    # num_win_x, num_win_o, num_draws, Games, Rewards = board_game_test(game, strategy_x,strategy_o_doucz)
    # game.print_test_to_file("gry_uczonej_X_z_douczana_O.txt",num_win_x, num_win_o, num_draws, Games, Rewards)


experiment_par_train()