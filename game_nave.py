# v 0020: moving the sensors down to read lower
# - punish * total_sensor
# - training 100000
# - reply memory 8000
# - batchSize 64
# - screen print n. of blocks avoided
import timeit
import matplotlib.pyplot as plt
import pygame
import time
import random
from pygame.color import THECOLORS
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.models import model_from_json


# Setting up the NN Architecture
train_frames = 100000#1000  # 150 1000000 # 30 * (minutes_train*60) #(60*hours_train) # fps * min2sec
observe = 8000      #120       # 40
buffer = observe    # 50000 # last actions took
batchSize = 64      # 12  #1 100 training set size
hours_train = 0.1
MAX_PANISH = -500.0
#minutes_train = 10
GAMMA = 0.9
NUM_INPUT = 5
NUM_OUTPUT = 3
NUM_FIRST_HLY = 32
NUM_SECOND_HLY = 64
model = Sequential()
model.add(Dense(NUM_FIRST_HLY, init='lecun_uniform', input_shape=(NUM_INPUT,)))
model.add(Activation('relu'))
model.add(Dense(NUM_SECOND_HLY, init='lecun_uniform'))
model.add(Activation('relu'))
model.add(Dense(NUM_OUTPUT, init='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

game_speed = 90     # doesn't work!
pygame.init()
GM_MODE = 2
display_width = 800
display_height = 600
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption('Nave ML')
nave_bm = pygame.image.load('nave_sm.png')
black = (0, 0, 0)
white = (255, 255, 255)
grey = (100, 100, 100)
light_grey = (150, 150, 150)
sensor_layers = 18        # num. of layers (resolution) on each sensor (x1 radar)
sensor_level_val = 0.1 #1.0  # resolution sensitivity value (x1)
max_reward_val = (sensor_layers * sensor_level_val) * 4    # max total sensitivity (x2)
flip_start_side = False
obs_start_state_x3 = 0
obst_speed = 25
dist_move = 40#10
filename = "model.h5"
obst_w = 400
obst_h = 300
OBST_W_HALF = obst_w / 2.0
DSP_W_HALF = display_width / 2.0
save_scores = []




def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()


def message_display(text):
    largeText = pygame.font.Font('freesansbold.ttf', 42)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.center = ((display_width / 2), (display_height / 2))
    screen.blit(TextSurf, TextRect)


def text_display(text, x, y, sz):
    #screen.fill(white)
    font = pygame.font.SysFont(None, sz)
    text = font.render(text, True, grey)
    screen.blit(text, (x, y))


def gameover():
    pygame.display.update()
   # time.sleep(0.5)


def draw_obstacles_update(x, y, obst_speed, thingw, thingh, color):
    new_y = y + obst_speed
    pygame.draw.rect(screen, color, [x, new_y, thingw, thingh])
    return new_y


def get_sensor_reading(reading):
    if reading == THECOLORS['black']:
        return 1.0
    else:
        return 0.0


def draw_nave_read_sensors(x, y, shw_sens):
    screen.blit(nave_bm, (x, y))
    y += 40     # radar base is a bit down
    # sensors
    shw_sens_local = False
    offset_l0 = 0
    offset_l45 = 0
    offset_c = 0
    offset_r = 0
    offset_r0 = 0
    sn_val_c = 0.0
    sn_val_l1 = 0.0
    sn_val_l2 = 0.0
    sn_val_r1 = 0.0
    sn_val_r2 = 0.0
    sens_data = []
    for s in range(sensor_layers):
        # left radar ang 0
        sen_base_x = int(x + 27 + offset_l0)  # centered
        if display_width > sen_base_x > 0: # if not out of the screen on LEFT
            sen_base_y = int(y)
            if get_sensor_reading(screen.get_at([sen_base_x, sen_base_y])) != 0:
                sn_val_l1 += sensor_level_val # got black. error div by 0?
            if shw_sens:
                pygame.draw.circle(screen, light_grey, (sen_base_x, sen_base_y), 2)
            offset_l0 -= 21
        else:
            sn_val_l1 += sensor_level_val # out of screen equal to black

        # left radar ang 45
        sen_base_x = int(x + 27 + offset_l45)
        if display_width > sen_base_x > 0:  # if not out of the screen on LEFT
            sen_base_y = int(y + offset_l45)
            if get_sensor_reading(screen.get_at([sen_base_x, sen_base_y])) != 0:
                sn_val_l2 += sensor_level_val  # got black. error div by 0?
            if shw_sens:
                pygame.draw.circle(screen, light_grey, (sen_base_x, sen_base_y), 2)
            offset_l45 -= 15
        else:
            sn_val_l2 += sensor_level_val  # out of screen equal to black

        # center radar ang 90
        sen_base_x = int(x + 32)  # centered
        if display_width > sen_base_x > 0:  # if not out of the screen on LEFT
            sen_base_y = int(y + offset_c)
            if get_sensor_reading(screen.get_at([sen_base_x, sen_base_y])) != 0:
                sn_val_c += sensor_level_val
                # sn_val_l1 += half_v  # half and half
                # sn_val_r1 += half_v
            if shw_sens:
                pygame.draw.circle(screen, light_grey, (sen_base_x, sen_base_y), 2)
            offset_c -= 20
        else:
            sn_val_c += sensor_level_val
            # sn_val_l1 += half_v  # half and half
            # sn_val_r1 += half_v

        # right radar ang 45
        sen_base_x = int(x + 37 + offset_r)
        if sen_base_x < display_width: # if not out of the screen on RIGHT
            sen_base_y = int(y - offset_r)
            if get_sensor_reading(screen.get_at([sen_base_x, sen_base_y])) != 0:
                sn_val_r1 += sensor_level_val # got black. error div by 0?
            if shw_sens:
                pygame.draw.circle(screen, light_grey, (sen_base_x, sen_base_y), 2)
            offset_r += 15
        else:
            sn_val_r1 += sensor_level_val  # out of screen equal to black

        # right radar ang 0
        sen_base_x = int(x + 37 + offset_r0)
        if sen_base_x < display_width:  # if not out of the screen on RIGHT
            sen_base_y = int(y)
            if get_sensor_reading(screen.get_at([sen_base_x, sen_base_y])) != 0:
                sn_val_r2 += sensor_level_val  # got black. error div by 0?
                # shw_sens_local = True
            if shw_sens:
                pygame.draw.circle(screen, light_grey, (sen_base_x, sen_base_y), 2)
            offset_r0 += 21
        else:
            sn_val_r2 += sensor_level_val  # out of screen equal to black

    pygame.draw.rect(screen, white, [10, 10, 200, 40]) # clean screen info data
    text_display("senL1: " + str(sn_val_l1), 10, 10, 24)
    text_display("senL2: " + str(sn_val_l2), 10, 30, 24)
    text_display("senC : " + str(sn_val_c), 10, 50, 24)
    text_display("senR1: " + str(sn_val_r1), 10, 70, 24)
    text_display("senR2: " + str(sn_val_r2), 10, 90, 24)
    sens_data.append(sn_val_l1)
    sens_data.append(sn_val_l2)
    sens_data.append(sn_val_c)
    sens_data.append(sn_val_r1)
    sens_data.append(sn_val_r2)
    return np.reshape(sens_data, (1, NUM_INPUT))


def human_move(x, y, ev):
    if x < 0 or x > display_width: # check out of bound
        return -1
    xc = 0
    if ev.type == pygame.KEYDOWN:
        if ev.key == pygame.K_LEFT:
            xc = -dist_move
        elif ev.key == pygame.K_RIGHT:
            xc = dist_move
    if ev.type == pygame.KEYUP:
        if ev.key == pygame.K_LEFT or ev.key == pygame.K_RIGHT:
            xc = 0

    return xc


def machine_move(x, y, act):
    if x < 0 or x > display_width - 50: # check out of bound
        return -1
    xc = 0
    if act == 0:    # move left
        xc = -dist_move
    elif act == 1:  # move none
        xc = xc
    elif act == 2:  # move right
        xc = dist_move
    return xc


def get_reward(xmov, st, x, y, obst_start_x, obst_start_y, obst_width, obst_height, nave_w):
    # out of screen
    #tts = np.abs(st.item(0) + st.item(1))
    tts = 0
    for s in range(len(st)):
    #for s in st:
        tts += np.abs(st.item(s))

    if xmov == -1:
        return MAX_PANISH #* tts
    # collision
    if y < obst_start_y+obst_height:
        if (x > obst_start_x and x < obst_start_x + obst_width):
            return MAX_PANISH #* tts
        if (x+nave_w > obst_start_x and x + nave_w < obst_start_x+obst_width):
            return MAX_PANISH# * tts
    return max_reward_val - tts  # sens max val 2.82


def process_minibatch(minibatch, model):
    """This does the heavy lifting, aka, the training. It's super jacked."""
    X_train = []
    y_train = []
    # Loop through our batch and create arrays for X and y
    # so that we can fit our model at every step.
    for memory in minibatch:
        # Get stored values.
        old_state_m, action_m, reward_m, new_state_m = memory
        # Get prediction on old state.
        old_qval = model.predict(old_state_m, batch_size=1)
        # Get prediction on new state.
        newQ = model.predict(new_state_m, batch_size=1)
        # Get our best move. I think?
        maxQ = np.max(newQ)
        y = np.zeros((1, 3))
        y[:] = old_qval[:]
        # Check for terminal state.
        if reward_m != -50:  # non-terminal state
            update = (reward_m + (GAMMA * maxQ))
        else:  # terminal state
            update = reward_m
            #message_display('GAME OVER!!')
        # Update the value for the action we took.
        y[0][action_m] = update
        X_train.append(old_state_m.reshape(NUM_INPUT,))
        y_train.append(y.reshape(NUM_OUTPUT,))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train


def start_game(_nave_w=None, obst_h=None, obst_w=None, _games=None, mode=None):
    if mode == 0:
        # 2 states: left or right screen
        global flip_start_side
        flip_start_side = not flip_start_side
        if flip_start_side:
            _obst_start_x = display_width - obst_w# * 0.5
        else:
            _obst_start_x = 0#-obst_w * 0.5
    elif mode == 1:
        # 3 states: left or right or center screen
        global obs_start_state_x3
        if obs_start_state_x3 == 0:
            _obst_start_x = display_width - OBST_W_HALF    # left
            obs_start_state_x3 = 1
        elif obs_start_state_x3 == 1:
            _obst_start_x = -OBST_W_HALF                   # right
            obs_start_state_x3 = 2
        else:
            _obst_start_x = display_width - OBST_W_HALF  - obst_w     # center
            obs_start_state_x3 = 0
    else:
        _obst_start_x = random.randrange(-obst_w/2.0, display_width-obst_w/2.0) # -display_width * .5

    return DSP_W_HALF - _nave_w * 0.5, _obst_start_x, -400  # 0 - obst_h/2


def save_nn_weights(_filename, _t):
    #model.save_weights(_filename + '-' + str(_t) + '.h5', overwrite=True)
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(_filename, overwrite=True)
    print("Saving model %s - %d" % (_filename, _t))


def training():
    model.compile(loss='mse', optimizer=RMSprop())
    clock = pygame.time.Clock()

    nave_w = 45
    pause = False
    show_sensors = True
    obst_swap = False
    obst_sawp_v = 400
    min_epsilon = 0.3
    # init nave location
    x_pos = (display_width * 0.49) - nave_w / 2
    y_pos = (display_height * 0.85)
    state = draw_nave_read_sensors(x_pos, y_pos, False)  # 1.41 max sensor val x 2
    obst_start_x = ((display_width/2) - obst_w/2)  # ((display_width/2) - obst_w/2)# + obst_sawp_v
    obst_start_y = -400
    obstacle_restarted = 0
    x_pos, obst_start_x, obst_start_y = start_game(nave_w, obst_h, obst_w, obstacle_restarted, GM_MODE)

    replay = [] # stores tuples of (S, A, R, S').
    epsilon = 1.0
    t = 0
    nave_distance = 0
    max_nave_distance = 0
    max_obstacle_restarted = 0
    start_time = timeit.default_timer()
    tot_time = 0
    fps = 0

    game_played = 0
    max_score_at_game = 0

    # Run the frames.
    while t < train_frames:
        # process user events
        for event in pygame.event.get():
            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_SPACE]:
                pause = not pause
                message_display('PAUSE')
            elif pressed[pygame.K_s]:
                show_sensors = not show_sensors
            elif pressed[pygame.K_q]:
                save_nn_weights(filename, t)
                pygame.quit()
                quit()
            # -Human control
            #x_move = human_move(x_pos, y_pos, event)
        if pause:
            continue

        t += 1
        nave_distance += 1 # my reward

        # * Choose an action.
        if random.random() < epsilon:
            action = random.randint(0, 2)  # EXPLORATION: choose random action
        else:
            qval = model.predict(state.reshape(1, NUM_INPUT), batch_size=1)
            action = (np.argmax(qval))  # EXPLOITATION: choose best action from Q(s,a) values

        x_move = machine_move(x_pos, y_pos, action)

        # clear screen and DRAW
        screen.fill(white)
        # draw Obstacles only moves on y (vertically)
        obst_start_y = draw_obstacles_update(obst_start_x, obst_start_y, obst_speed, obst_w, obst_h, black)
        # check if need to reset obstacles start position
        if obst_start_y > display_height:
            #obst_start_y = 0 - obst_h / 2
            #obst_start_x = (display_width / 2)  # random.randrange(-display_width*.5, display_width)
            obstacle_restarted += 1
            _, obst_start_x, obst_start_y = start_game(nave_w, obst_h, obst_w, obstacle_restarted, GM_MODE)

        # draw Nave
        if x_move != -1: # check out of screen
            new_state = draw_nave_read_sensors(x_pos + x_move, y_pos, show_sensors)  # 1.41 max sensor val x 2

        # * Observe reward
        reward = get_reward(x_move, new_state, x_pos, y_pos, obst_start_x, obst_start_y, obst_w, obst_h, nave_w)

        # * Experience replay storage.
        replay.append((state, action, reward, new_state))

        # If we're done observing, start training.
        if t > observe:
            # If we've stored enough in our buffer, pop the oldest.
            if len(replay) > buffer:
                replay.pop(0)

            # Randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)

            # Get training values.
            X_train, y_train = process_minibatch(minibatch, model)

            # Train the model on this batch.
            #history = LossHistory()
            model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=1)
            #loss_log.append(history.losses)
        else:
            message_display('memory: ' + str(t)+'/'+str(observe))

        # Update the starting state with S'.
        state = new_state
        print('steps: ' + str(t) + '/' + str(train_frames))

        # Decrement epsilon over time.
        if epsilon > 0.1 and t > observe:
            epsilon -= (1.0 / train_frames)

        # We died, so update stuff.
        if reward < 0:
            save_scores.append(nave_distance)
            #save_scores.append([t, nave_distance])
            game_played += 1
            message_display('GAME OVER!!')
            x_pos, obst_start_x, obst_start_y = start_game(nave_w, obst_h, obst_w, obstacle_restarted, GM_MODE)

            # Update max.
            if nave_distance > max_nave_distance:
                max_nave_distance = nave_distance
                max_score_at_game = game_played-1
            # Update max objects passed.
            if obstacle_restarted > max_obstacle_restarted:
                max_obstacle_restarted = obstacle_restarted

            # Time it.
            tot_time = timeit.default_timer() - start_time
            fps = nave_distance / tot_time

            # Output some stuff so we can watch.
            print("Max: %d at %d\tepsilon %f\t(%d)\t%f fps" %
                  (max_nave_distance, t, epsilon, nave_distance, fps))

            # Reset.
            nave_distance = 0
            obstacle_restarted = 0
            start_time = timeit.default_timer()
        else:
            x_pos += x_move

        # Save the model every 25,000 frames.
        if t % (train_frames/20) == 0:
            save_nn_weights(filename, t)

        text_display("reward: " + str(reward), 10, 120, 24)
        text_display("steps: " + str(t) + '/' + str(train_frames)+  " *SCORE: " +
                     str(nave_distance) + "/" + str(max_nave_distance) + "@" + str(max_score_at_game)
                     + "/" + str(game_played), 350, 10, 24)
        text_display("epsilon: " + str(epsilon)+"  BksAv: " +
                     str(obstacle_restarted)+'/' + str(max_obstacle_restarted), 350, 30, 24)
        # print('reward: ' + str(reward))
        # print('action: ' + str(action))
        print('est_time: ' + str((tot_time * train_frames)/60))
        print('fps: ' + str(fps))
        if not pause:
            pygame.display.update()
            clock.tick(game_speed)

    #plt.scatter(save_scores[:0], save_scores[:1])
    plt.scatter(np.arange(game_played), save_scores)
    plt.show()


def test_ml(init, test_games):
    clock = pygame.time.Clock()
    nave_w = 45
    obst_w = 400
    obst_h = 300
    pause = False
    show_sensors = True
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    nave_distance = 0
    max_nave_distance = 0
    game_played = 0
    max_score_at_game = 0

    # load weights into new model
    loaded_model.load_weights(filename)
    print("......Loaded model from disk")

    for tg in range(test_games):
        print("Game: %s" % (tg))
        game_over = False
        # init obstacles location
        x_pos = (display_width * 0.49) - nave_w/2
        y_pos = (display_height * 0.85)

        x_pos, obst_start_x, obst_start_y = start_game(nave_w, obst_h, obst_w, None, GM_MODE)

        #obst_start_x = ((display_width/2) - obst_w/2)# + obst_sawp_v
        obst_start_y = -400
        state = draw_nave_read_sensors(x_pos, y_pos, False)  # 1.41 max sensor val x 2
        # init hyper parameter
        reward = 0.0
        score = 0
        obstacle_restarted = 0
        while not game_over:
            # process user events
            for event in pygame.event.get():
                pressed = pygame.key.get_pressed()
                if pressed[pygame.K_SPACE]:
                    pause = not pause
                    message_display('PAUSE')
                elif pressed[pygame.K_s]:
                    show_sensors = not show_sensors
                elif pressed[pygame.K_q]:
                    pygame.quit()
                    quit()
                # -Human control
                #x_move = human_move(x_pos, y_pos, event)
            if pause:
                continue

            nave_distance += 1  # my reward

            qval = loaded_model.predict(state.reshape(1, NUM_INPUT), batch_size=1)
            print qval
            action = (np.argmax(qval))  # take action with highest Q-value
            #print('Move #: %s; Taking action: %s' % (i, action))
            # state = makeMove(state, action)
            x_move = machine_move(x_pos, y_pos, action)



            # clear screen and DRAW
            screen.fill(white)
            # draw Obstacles only moves on y (vertically)
            obst_start_y = draw_obstacles_update(obst_start_x, obst_start_y, obst_speed, obst_w, obst_h, black)
            # check if need to reset obstacles start position
            if obst_start_y > display_height:
                #obst_start_y = 0 - obst_h
                obstacle_restarted += 1
                # obst_start_x = random.randrange(0, display_width)
                #obst_start_x = display_width / 2  # display_width / 2 x
                _, obst_start_x, obst_start_y = start_game(nave_w, obst_h, obst_w, None, GM_MODE)

            # draw Nave
            state = draw_nave_read_sensors(x_pos+x_move, y_pos, show_sensors) # 1.41 max sensor val x 2

            # reward = getReward(state)
            reward = get_reward(x_move, state, x_pos, y_pos, obst_start_x, obst_start_y, obst_w, obst_h, nave_w)
            if reward < 0:  # crashed with obst or screen borders
                message_display('GAME OVER!!')
                game_over = True
                update = reward  # terminal state
                game_played += 1
                # Update max.
                if nave_distance > max_nave_distance:
                    max_nave_distance = nave_distance
                    max_score_at_game = game_played - 1
                nave_distance = 0
            else:
                x_pos += x_move  # update final nave position
                score += 1

            text_display("reward: " + str(reward), 10, 120, 24)
            text_display("*SCORE: " + str(nave_distance) + "/" +
                         str(max_nave_distance) + "@" + str(max_score_at_game)
                         + "/" + str(game_played), 450, 10, 24)
            if not pause:
                pygame.display.update()
                clock.tick(game_speed)

    # N = 50
    # x = np.random.rand(N)
    # y = np.random.rand(N)
    #
    # plt.scatter(x, y)
    # plt.show()


#training()
test_ml(1, 30)

# keep_playing = True
# while keep_playing:
#     test_ml(1, 30)
#     for event in pygame.event.get():
#         pressed = pygame.key.get_pressed()
#         if pressed[pygame.K_SPACE]:
#             pause = not pause
#             message_display('PAUSE')
#         elif pressed[pygame.K_s]:
#             show_sensors = not show_sensors
#         #elif pressed[pygame.K_r]:
#         elif pressed[pygame.K_q]:
#             keep_playing = False
#             pygame.quit()
#             quit()
#             # -Human control
#             # x_move = human_move(x_pos, y_pos, event)
#     if pause:
#         continue


# pygame.quit()
# quit()




















