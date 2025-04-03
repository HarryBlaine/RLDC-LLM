# from maze_env import Maze
import pickle
from stock_single_asset_disagree import stock
from PPO import PPO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from torch.distributions import Categorical
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def BackTest(env, model, show_log=True, my_trick=False):
    model.eval()
    observation = env.reset()
    rewards = 0
    h_out = (torch.zeros([1, 1, 256], dtype=torch.float).to(device), torch.zeros([1, 1, 256], dtype=torch.float).to(device))
    step = 0
    while True:
        h_in = h_out
        prob, h_out, _ = model.pi(torch.from_numpy(observation).float(), h_in)
        prob = prob.view(-1)
        action = torch.argmax(prob).item()
        observation_, reward, done, _ = env.step(action, show_log=False)
        rewards = rewards + reward
        observation = observation_
        # break while loop when end of this episode
        if done:
            break
        step += 1
    print('Test total_profit:%.3f' % (env.total_profit))
    model.train()
    return env, rewards


if __name__ == "__main__":
    max_round = 5001

    file_path = 'Data/CL.NYM_processed.csv'
    df = pd.read_csv(file_path)
    df = df.reset_index(drop=True)  # 去除前几天没有均线信息

    df_append = pd.read_csv("Data/887637.WI_processed.csv")
    df["open"] = df_append["open"]
    df["high"] = df_append["high"]
    df["low"] = df_append["low"]
    df["close"] = df_append["close"]
    df["volume"] = df_append["volume"]

    df["Real_close"] = df["close_o"]
    df_append = pd.read_csv("Data/oil_with_solar_cross_disagreement_aligned.csv")
    # df["use"] = df_append["avg_sentiment"]
    df["use"] = df_append["CrossDisagreement"]

    df_append = pd.read_csv("Data/oil_with_basic_disagreement_processed.csv")
    df["basic_disagreement"] = df_append["basic_disagreement"]

    df_train = df.iloc[0:2000]
    df_test = df.iloc[2000:]

    scaler = StandardScaler()

    columns_to_normalize = ['open_o', 'high_o', 'low_o', 'close_o', 'volume_o', "open", "high", "low", "close", "volume","use","basic_disagreement"]

    df_train[columns_to_normalize] = scaler.fit_transform(df_train[columns_to_normalize])
    df_test[columns_to_normalize] = scaler.transform(df_test[columns_to_normalize])

    env_train = stock(df_train)
    env_test = stock(df_test)

    model = PPO(env_train.n_features, env_train.n_actions)
    step = 0
    training_profit = []
    testing_profit = []

    training_reward = []
    testing_reward = []

    train_max = 0
    test_max = 0

    for episode in range(max_round):
        # initial observation
        rewards = 0
        buy_action = 0
        sell_action = 0
        not_hold_action = 0
        hold_action = 0

        h_out = (torch.zeros([1, 1, 256], dtype=torch.float).to(device), torch.zeros([1, 1, 256], dtype=torch.float).to(device))
        observation = env_train.reset()
        step = 0
        while True:
            h_in = h_out
            prob1, h_out, logit = model.pi(torch.from_numpy(observation).float(), h_in)
            # logits = (logit / temperature)
            # prob = torch.softmax(logits, dim=2).view(-1)
            prob = prob1.view(-1)

            m = Categorical(prob)
            action = m.sample().item()
            # print(action)
            observation_, reward, done, action_state = env_train.step(action, show_log=False)
            rewards = rewards + reward
            model.put_data((observation, action, reward, observation_, prob[action].item(), h_in, h_out, done))
            observation = observation_
            # break while loop when end of this episode
            step += 1
            # if (step % 256 == 0 or done) and episode >= 5:
            #     ETH_model.train_net()
            if action_state == 0:
                sell_action+=1
            elif action_state == 1:
                hold_action+=1
            elif action_state == 2:
                not_hold_action+=1
            else:
                buy_action+=1
            if done:
                break

        if episode >= 5:
            model.train_net()
            # if temperature > 1:
            #     temperature-= 0.01
        print('epoch:%d, buy_action: %d,  sell_action: %d, hold_action: %d, not_hold_action: %d,  total_profit:%.3f' % (episode, buy_action, sell_action, hold_action,not_hold_action,
                                                                                                                        env_train.total_profit))

        # print('EEEENV : , buy_action: %d,  sell_action: %d, hold_action: %d, not_hold_action: %d' % (env_train.buy_action, env_train.sell_action, env_train.hold_action, env_train.not_hold_action))

        training_profit.append(env_train.total_profit)
        training_reward.append(rewards)

        if episode % 10 == 0:
            model_name = 'Models/' + str(episode) + '.pkl'
            plt.clf()
            pickle.dump(model, open(model_name, 'wb'))

        env_test, test_rewards = BackTest(env_test,model, show_log=False)
        testing_profit.append(env_test.total_profit)
        testing_reward.append(test_rewards)
        print('Test 8876_Reward :%.3f' % test_rewards)



