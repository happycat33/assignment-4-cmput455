
import pexpect

# path to the two players
# player 1 plays first
player1='random_player/nogo_random.py'
player2='nogo4/nogo4.py'

# stats for win/timeout
# note that timeout is considered lost
win1=0
win2=0
numTimeout=0
# time limit per move
TIMEOUT=30
SAFETY_MARGIN=1

def getMove(p,color):
    '''
    send genmove command to the players
    used to play a move
    '''
    p.sendline('genmove '+color)
    p.expect([pexpect.TIMEOUT,'= [a-zA-Z][0-9]','= resign'])
    if p.after==pexpect.TIMEOUT:
        return 'timeout'
    return p.after.decode("utf-8")[2:]

def playMove(p,color,move):
    '''
    send play command to the players
    used to sync the board
    '''
    p.sendline('play '+color+' '+move)

def setupPlayer(p):
    '''
    configure the players
    '''
    p.sendline('boardsize 7')
    p.sendline('clear_board')
    p.sendline('timelimit {}'.format(TIMEOUT))

def playSingleGame(alternative=False):
    '''
    play a single game
    return the winner and the number of timeouts
    '''
    # alternating the first and second player on each new game
    if not alternative:
        p1=pexpect.spawn('python3 '+player1,timeout=TIMEOUT+SAFETY_MARGIN)
        p2=pexpect.spawn('python3 '+player2,timeout=TIMEOUT+SAFETY_MARGIN)
    else:
        p1=pexpect.spawn('python3 '+player2,timeout=TIMEOUT+SAFETY_MARGIN)
        p2=pexpect.spawn('python3 '+player1,timeout=TIMEOUT+SAFETY_MARGIN)

    # observer player
    # monitor the game state
    ob=pexpect.spawn('python3 random_player/nogo_random.py')
    # set up the players
    setupPlayer(p1)
    setupPlayer(p2)
    result=None
    istimeout=0
    sw=0
    while 1:
        if sw==0:
            move=getMove(p1,'b')
            assert(move!='pass')
            if move=='resign':
                result=2
                break
            elif move=='timeout':
                result=2
                istimeout=1
                break
            playMove(p2,'b',move)
            playMove(ob,'b',move)
        else:
            move=getMove(p2,'w')
            assert(move!='pass')
            if move=='resign':
                result=1
                break
            elif move=='timeout':
                result=1
                istimeout=1
                break
            playMove(p1,'w',move)
            playMove(ob,'w',move)

        # player 1 cases
        if (sw == 0 and not alternative) or (sw == 1 and alternative):
            print(f'Player1 made move: {move}')
        # player 2 otherwise
        else:
            print(f'Player2 made move: {move}')

        # switch turn
        sw=1-sw

        # check game status
        ob.sendline('gogui-rules_final_result')
        ob.expect(['= black','= white','= unknown'])
        status=ob.after.decode("utf-8")[2:]

        if status=='black':
            result=1
            break
        elif status=='white':
            result=2
            break
        else:
            assert(status=='unknown')

    if result==1 and alternative==False or result==2 and alternative==True:
        winner = 'Player1'
    else:
        assert(result==1 and alternative==True or result==2 and alternative==False)
        winner = 'Player2'

    # print result
    print(f'Winner: {winner} Timeout: {istimeout}\n')
    return result,istimeout

def playGames(numGame=1):
    '''
    play the specified number of games with alternating turns
    '''
    global win1,win2,numTimeout

    for i in range(0,numGame):
        if(i<numGame/2):
            alter=False
        else:
            alter=True
        result,t=playSingleGame(alternative=alter)
        numTimeout+=t
        assert result==1 or result==2
        if result==1 and alter==False or result==2 and alter==True:
            win1+=1
        else:
            assert(result==1 and alter==True or result==2 and alter==False)
            win2+=1

def outputResult():
    print('Player1 wins',win1,'Player2 wins',win2, 'Timeouts', numTimeout)

def saveResult():
    f = open("game_results.txt", "w")
    f.write("player 1: {}\n".format(player1))
    f.write("player 2: {}\n".format(player2))
    f.write("player 1 wins: {}\n".format(win1))
    f.write("player 2 wins: {}\n".format(win2))
    f.close()

playGames()
outputResult()
saveResult()
