
import pickle
from stock_env_all_disagree import stock
from PPO import PPO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from torch.distributions import Categorical
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


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

    df_dy = pd.read_csv("Data/dy_Connectedness_Index.csv")

    coal_arg = df_dy['Coal_return']
    # coal_arg = 0.029863124
    df_append = pd.read_csv("Data/ATW.IPE_processed.csv")
    df["open_Coal"] = df_append["open"] * coal_arg
    df["high_Coal"] = df_append["high"] * coal_arg
    df["low_Coal"] = df_append["low"] * coal_arg
    df["close_Coal"] = df_append["close"] * coal_arg
    df["volume_Coal"] = df_append["volume"] * coal_arg
    df_append = pd.read_csv("Data/oil_with_coal_cross_disagreement_aligned.csv")
    df["use_Coal"] = df_append["avg_sentiment"]
    # df["use_Coal"] = df_append["CrossDisagreement"]

    # cooper_arg = 0.068308634
    cooper_arg = df_dy['Copper_return']
    df_append = pd.read_csv("Data/HG.CMX_processed.csv")
    df["open_Copper"] = df_append["open"] * cooper_arg
    df["high_Copper"] = df_append["high"] * cooper_arg
    df["low_Copper"] = df_append["low"] * cooper_arg
    df["close_Copper"] = df_append["close"] * cooper_arg
    df["volume_Copper"] = df_append["volume"] * cooper_arg
    df_append = pd.read_csv("Data/oil_with_copper_cross_disagreement_aligned.csv")
    df["use_Copper"] = df_append["avg_sentiment"]
    # df["use_Copper"] = df_append["CrossDisagreement"]

    # TU_arg = 0.031753223
    TU_arg = df_dy['2ybond_return']
    df_append = pd.read_csv("Data/TU_CBT_processed.csv")
    df["open_TU"] = df_append["open"] * TU_arg
    df["high_TU"] = df_append["high"] * TU_arg
    df["low_TU"] = df_append["low"] * TU_arg
    df["close_TU"] = df_append["close"] * TU_arg
    df["volume_TU"] = df_append["volume"] * TU_arg
    df_append = pd.read_csv("Data/oil_with_treasurybond_cross_disagreement_aligned.csv")
    df["use_TU"] = df_append["avg_sentiment"]
    # df["use_TU"] = df_append["CrossDisagreement"]

    # TY_arg = 0.0116677355
    TY_arg = df_dy['10ybond_return']
    df_append = pd.read_csv("Data/TY_CBT_processed.csv")
    df["open_TY"] = df_append["open"] * TY_arg
    df["high_TY"] = df_append["high"] * TY_arg
    df["low_TY"] = df_append["low"] * TY_arg
    df["close_TY"] = df_append["close"] * TY_arg
    df["volume_TY"] = df_append["volume"] * TY_arg
    df_append = pd.read_csv("Data/oil_with_treasurybond_cross_disagreement_aligned.csv")
    df["use_TY"] = df_append["avg_sentiment"]
    # df["use_TY"] = df_append["CrossDisagreement"]

    # DJI_arg = 0.51270205
    DJI_arg = df_dy['Dow Jones_return']
    df_append = pd.read_csv("Data/DJI.GI_processed.csv")
    df["open_DJI"] = df_append["open"] * DJI_arg
    df["high_DJI"] = df_append["high"] * DJI_arg
    df["low_DJI"] = df_append["low"] * DJI_arg
    df["close_DJI"] = df_append["close"] * DJI_arg
    df["volume_DJI"] = df_append["volume"] * DJI_arg
    df_append = pd.read_csv("Data/oil_with_dowjones_cross_disagreement_aligned.csv")
    df["use_DJI"] = df_append["avg_sentiment"]
    # df["use_DJI"] = df_append["CrossDisagreement"]

    # gold_arg = 0.005807469
    gold_arg = df_dy['Gold_return']
    df_append = pd.read_csv("Data/GC.CMX_processed.csv")
    df["open_Gold"] = df_append["open"] * gold_arg
    df["high_Gold"] = df_append["high"] * gold_arg
    df["low_Gold"] = df_append["low"] * gold_arg
    df["close_Gold"] = df_append["close"] * gold_arg
    df["volume_Gold"] = df_append["volume"] * gold_arg
    df_append = pd.read_csv("Data/oil_with_gold_cross_disagreement_aligned.csv")
    df["use_Gold"] = df_append["avg_sentiment"]
    # df["use_Gold"] = df_append["CrossDisagreement"]

    # gas_arg = 0.044704911
    gas_arg = df_dy['Natural gas_return']
    df_append = pd.read_csv("Data/NG.NYM_processed.csv")
    df["open_gas"] = df_append["open"] * gas_arg
    df["high_gas"] = df_append["high"] * gas_arg
    df["low_gas"] = df_append["low"] * gas_arg
    df["close_gas"] = df_append["close"] * gas_arg
    df["volume_gas"] = df_append["volume"] * gas_arg
    df_append = pd.read_csv("Data/oil_with_gas_cross_disagreement_aligned.csv")
    df["use_gas"] = df_append["avg_sentiment"]
    # df["use_gas"] = df_append["CrossDisagreement"]

    # Nasdaq_arg = 0.263105224
    Nasdaq_arg = df_dy['NASDAQ_return']
    df_append = pd.read_csv("Data/IXIC.GI_processed.csv")
    df["open_Nasdaq"] = df_append["open"] * Nasdaq_arg
    df["high_Nasdaq"] = df_append["high"] * Nasdaq_arg
    df["low_Nasdaq"] = df_append["low"] * Nasdaq_arg
    df["close_Nasdaq"] = df_append["close"] * Nasdaq_arg
    df["volume_Nasdaq"] = df_append["volume"] * Nasdaq_arg
    df_append = pd.read_csv("Data/oil_with_nasdaq_cross_disagreement_aligned.csv")
    df["use_Nasdaq"] = df_append["avg_sentiment"]
    # df["use_Nasdaq"] = df_append["CrossDisagreement"]

    # solar_arg = 0.016456455
    solar_arg = df_dy['Solar_return']
    df_append = pd.read_csv("Data/887637.WI_processed.csv")
    df["open_solar"] = df_append["open"] * solar_arg
    df["high_solar"] = df_append["high"] * solar_arg
    df["low_solar"] = df_append["low"] * solar_arg
    df["close_solar"] = df_append["close"] * solar_arg
    df["volume_solar"] = df_append["volume"] * solar_arg
    df_append = pd.read_csv("Data/oil_with_solar_cross_disagreement_aligned.csv")
    df["use_solar"] = df_append["avg_sentiment"]
    # df["use_solar"] = df_append["CrossDisagreement"]

    # wind_arg = 0.013436089
    wind_arg = df_dy['Wind_return']
    df_append = pd.read_csv("Data/wind_aligned.csv")
    df["close_wind"] = df_append["close"] * wind_arg
    df_append = pd.read_csv("Data/oil_with_wind_cross_disagreement_aligned.csv")
    df["use_wind"] = df_append["avg_sentiment"]
    # df["use_wind"] = df_append["CrossDisagreement"]
    df_append = pd.read_csv("Data/oil_with_basic_disagreement_processed.csv")
    df["basic_disagreement"] = df_append["basic_disagreement"]
    df["Real_close"] = df["close_o"]

    df_train = df.iloc[0:1520]
    df_test = df.iloc[1520:1770]

    scaler = StandardScaler()

    columns_to_normalize = ['open_o', 'high_o', 'low_o', 'close_o', 'volume_o',
                            'open_Coal', 'high_Coal', 'low_Coal', 'close_Coal', 'volume_Coal',"use_Coal",
                            'open_Copper', 'high_Copper', 'low_Copper', 'close_Copper',"use_Copper",
                            'volume_Copper', 'open_TU', 'high_TU', 'low_TU', 'close_TU', 'volume_TU', "use_TU",
                            'open_TY', 'high_TY', 'low_TY', 'close_TY', 'volume_TY', "use_TY",
                            'open_DJI', 'high_DJI', 'low_DJI', 'close_DJI', 'volume_DJI',  "use_DJI",
                            'open_Gold', 'high_Gold', 'low_Gold', 'close_Gold', 'volume_Gold', "use_Gold",
                            'open_gas', 'high_gas', 'low_gas', 'close_gas', 'volume_gas',  "use_gas",
                            'open_Nasdaq', 'high_Nasdaq', 'low_Nasdaq', 'close_Nasdaq', 'volume_Nasdaq', "use_Nasdaq",
                            'open_solar', 'high_solar', 'low_solar', 'close_solar', 'volume_solar', "use_solar",
                            'close_wind',"use_wind","basic_disagreement"]


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
        print('Test ALL_Reward :%.3f' % test_rewards)



