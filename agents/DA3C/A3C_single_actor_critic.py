"""
双重异步优势演员评论家算法
一个网络多个forward函数+总损失函数
"""
import copy
import pickle
import random
import time
import numpy as np
import torch
from torch import multiprocessing
from torch.multiprocessing import Queue
from torch.optim import Adam
from agents.Base_Agent import Base_Agent
from utilities.Utility_Functions import create_actor_distribution, SharedAdam
from utilities.data_structures.Config import Config
from environments.SO_DFJSP import SO_DFJSP_Environment
import torch.nn.functional as F
from torch import nn
from utilities.Utility_Class import MyError
from visdom import Visdom

# 监控训练过程
vis = Visdom()
win = 'single_actor_critic'
vis.line(X=[0], Y=[0], win=win, opts=dict(title='Single actor critic', xlabel='epoch', ylable='delay_time', xmin=30))


# 构建工序策略网络类
class ActorCriticNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, output_size, output_size_1, output_size_2):
        super(ActorCriticNet, self).__init__()
        # 网络输入层
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size), nn.ReLU()])
        # 定义网络隐藏层
        for i in range(hidden_layer - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        # 定义评论家网络输出层
        self.layers_0 = nn.Linear(hidden_size, output_size)
        # 定义工序策略网络输出层
        self.layers_1 = nn.Linear(hidden_size, output_size_1)
        # 定义机器策略网络输出层
        self.layers_2 = nn.Linear(hidden_size, output_size_2)

    def forward_task(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.layers_1(x)
        x = F.softmax(x, dim=-1)
        return x

    def forward_machine(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.layers_2(x)
        x = F.softmax(x, dim=-1)
        return x

    def forward_critic(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.layers_0(x)
        return x

class DA3C(Base_Agent, Config):
    """Actor critic A3C algorithm"""
    agent_name = "A3C"
    def __init__(self):
        Base_Agent.__init__(self)  # 继承基础智能体类
        Config.__init__(self)  # 继承算法超参数类
        self.num_processes = multiprocessing.cpu_count()  # 电脑线程数量|四核八线程
        self.worker_processes = max(1, self.num_processes - 8)  # 启用线程数
        self.path = 'D:\Python project\Deep_Reinforcement_Learning_FJSP\data\generated'  # 测试算例的存储位置
        self.file_name = 'DDT1.0_M15_S1'  # 测试算例的文件夹名字
        self.test_environment = SO_DFJSP_Environment(use_instance=False, path=self.path, file_name=self.file_name)  # 测试环境
        # 超参数
        self.learning_rate = self.hyper_parameters["DA3C"]["learning_rate"]  # 学习率
        self.discount_rate = self.hyper_parameters["DA3C"]["discount_rate"]  # 折扣率
        self.num_episodes_to_run = self.hyper_parameters["DA3C"]["num_episodes_to_run"]  # 总的步数
        self.epsilon_decay_rate_denominator = self.hyper_parameters["DA3C"]["epsilon_decay_rate_denominator"]
        # 初始化锁对象 用来更新全局网络参数
        self.optimizer_lock = multiprocessing.Lock()   # 创建锁对象，用于更新全局网络参数
        # 定义策略网络、评论家网络、优化器
        self.actor_critic = ActorCriticNet(input_size=20, hidden_size=200, hidden_layer=5, output_size=1,
                                           output_size_1=6, output_size_2=5)
        self.actor_critic_optimizer = SharedAdam(self.actor_critic.parameters(), lr=self.learning_rate, eps=1e-4)

    def run_n_episodes(self):
        """运行环境n次直到完成，然后总结结果并保存模型(如果要求的话)"""
        start = time.time()
        gradient_updates_queue = Queue()
        episode_number = multiprocessing.Value('i', 0)  # 多线程共享整数型+初始值为0的参数
        episodes_per_process = int(self.num_episodes_to_run / self.worker_processes) + 1
        processes = []  # 初始化线程列表
        self.actor_critic.share_memory()
        self.actor_critic_optimizer.share_memory()
        optimizer_worker = multiprocessing.Process(target=self.update_shared_model, args=(gradient_updates_queue,))

        optimizer_worker.start()  # 启动总梯度更新主线程
        for process_num in range(self.worker_processes):
            worker = Actor_Critic_Worker(process_num, self.actor_critic, episode_number, self.optimizer_lock,
                                         self.actor_critic_optimizer, self.hyper_parameters, episodes_per_process,
                                         copy.deepcopy(self.actor_critic), gradient_updates_queue, self.test_environment)
            worker.start()  # 启动各子线程run()函数
            processes.append(worker)
        for worker in processes:
            worker.join()  # 让子线程结束后主线程再结束
        optimizer_worker.terminate()  # 主线程退出
        time_taken = time.time() - start
        self.save_actor_model(self.save_model)  # 保存训练好的策略网络
        return time_taken

    def save_actor_model(self, save_model_boole=False):
        """保存全局策略网络"""
        if save_model_boole:
            torch.save(self.actor_critic.state_dict(), 'actor_critic_net.ckpt')
        else:
            print("该训练过程未保存模型")
        return None

    def update_shared_model(self, gradient_updates_queue):
        """收到工作线程的梯度{信息传入队列}，更新全局网络梯度"""
        while True:
            gradients = gradient_updates_queue.get()
            with self.optimizer_lock:
                # 更新工序策略网络梯度
                self.actor_critic_optimizer.zero_grad()
                for grads, params in zip(gradients, self.actor_critic.parameters()):
                    params._grad = grads.detach().clone()  # maybe need to do grads.clone()  # 子线程梯度值传递给全局网络参数
                self.actor_critic_optimizer.step()  # 依据传递的新的梯度值更新参数

class Actor_Critic_Worker(torch.multiprocessing.Process):
    """演员评论工作者将玩游戏的指定集数 """
    def __init__(self, worker_num, actor_critic, counter, optimizer_lock, actor_critic_optimizer, hyper_parameter,
                 episodes_to_run, local_actor_critic_model, gradient_updates_queue, test_environment):
        torch.multiprocessing.Process.__init__(self)
        self.test_environment = test_environment  # 测试环境
        self.environment_train = None  # 初始化环境对象
        self.action_types = self.test_environment.action_types  # 动作类型
        self.worker_num = worker_num  # 线程数
        self.gradient_clipping_norm = hyper_parameter["DA3C"]["gradient_clipping_norm"]  # 梯度裁剪值
        self.discount_rate = hyper_parameter["DA3C"]["discount_rate"]  # 折扣率
        self.exploration_worker_difference = hyper_parameter["DA3C"]["exploration_worker_difference"]
        self.epsilon_decay_denominator = hyper_parameter["DA3C"]["epsilon_decay_rate_denominator"]
        self.normalise_rewards = False  # 标准化回报
        self.actions_size = self.test_environment.actions_size  # 二维离散动作[6, 5]
        self.actor_critic_model = actor_critic  # 工序策略网络
        self.local_actor_critic_model = local_actor_critic_model  # 局部工序策略网络
        self.local_actor_critic_optimizer = Adam(self.local_actor_critic_model.parameters(), lr=0.0, eps=1e-4)
        self.counter = counter
        self.optimizer_lock = optimizer_lock
        self.actor_critic_optimizer = actor_critic_optimizer
        self.episodes_to_run = episodes_to_run  # 周期总数
        self.episode_number = 0  # 子线程周期数
        self.gradient_updates_queue = gradient_updates_queue
        self.episode_states = []  # 状态列表
        self.episode_actions = []  # 动作列表
        self.episode_rewards = []  # 回报列表
        self.episode_log_action_task_probabilities = []  # 动作log概率列表
        self.episode_log_action_machine_probabilities = []  # 机器策略网络动作log概率列表
        self.critic_outputs = []  # 评论家输出的V值列表

    def generated_new_environment(self):
        """返回新环境对象"""
        DDT = random.uniform(0.5, 1.5)
        M = random.randint(10, 20)
        S = random.randint(1, 1)
        return SO_DFJSP_Environment(use_instance=True, DDT=DDT, M=M, S=S)

    def run(self):
        """开启工作线程"""
        torch.set_num_threads(1)
        for ep_ix in range(self.episodes_to_run):
            self.environment_train = self.test_environment
            with self.optimizer_lock:  # 锁定网络更新线程网络参数
                Base_Agent.copy_model_over(self.actor_critic_model, self.local_actor_critic_model)
            epsilon_exploration = self.calculate_new_exploration()  # 计算新的探索参数
            state = self.environment_train.reset()  # 初始化状态
            done = False
            self.episode_states = []  # 状态列表
            self.episode_actions = []  # 动作列表
            self.episode_rewards = []  # 回报列表
            self.episode_log_action_task_probabilities = []  # 工序策略网络动作log概率列表
            self.episode_log_action_machine_probabilities = []  # 机器策略网络动作log概率列表
            self.critic_outputs = []  # 评论家输出的V值列表
            # 采样一条轨迹
            while not done:
                action_task, action_task_log_prob = self.pick_action_and_log_prob(self.local_actor_critic_model, state,
                                                                                  epsilon_exploration=0, forward="task")
                action_machine, action_machine_log_prob = self.pick_action_and_log_prob(self.local_actor_critic_model,
                                                                                        state, epsilon_exploration=0,
                                                                                        forward="machine")
                actions = np.array([action_task, action_machine])  # 二维离散动作
                critic_outputs = self.get_critic_value(self.local_actor_critic_model, state)
                next_state, reward, done = self.environment_train.step(actions)
                self.episode_states.append(state)
                self.episode_actions.append(actions)
                self.episode_rewards.append(reward)
                self.episode_log_action_task_probabilities.append(action_task_log_prob)
                self.episode_log_action_machine_probabilities.append(action_machine_log_prob)
                self.critic_outputs.append(critic_outputs)
                state = next_state

            total_loss = self.calculate_total_loss()
            self.put_gradients_in_queue(total_loss)
            self.episode_number += 1
            vis.line(X=[self.counter.value], Y=[self.environment_train.delay_time_sum], win=win, update='append')
            # 每间隔10个周期运行一次测试算例并动态绘制目标值曲线
            with self.counter.get_lock():
                self.counter.value += 1
                # print("运行总步数：", self.counter.value)
                # if self.counter.value % 5 == 0:
                #     state = self.test_environment.reset()
                #     while not self.test_environment.done:
                #         action_task, action_task_log_prob \
                #             = self.pick_action_and_log_prob(self.local_actor_critic_model, state, epsilon_exploration,
                #                                             forward="task")
                #         state_add = np.append(state, action_task)  # 带选择的工序规则信息的状态
                #         action_machine, action_machine_log_prob \
                #             = self.pick_action_and_log_prob(self.local_actor_critic_model, state_add,
                #                                             epsilon_exploration, forward="machine")
                #         actions = np.array([action_task, action_machine])  # 二维离散动作
                #         next_state, reward, done = self.test_environment.step(actions)
                #         state = next_state
                #     print("累计回报:", self.test_environment.reward_sum)
                #     print("实际延期时间：", self.test_environment.delay_time_sum)
                #     print("运行测试算例并更新目标值曲线")

    def calculate_new_exploration(self):
        """计算新的勘探参数。它在当前的上下3X范围内随机选取一个点"""
        with self.counter.get_lock():
            epsilon = 1.0 / (1.0 + (self.counter.value / self.epsilon_decay_denominator))
        epsilon = max(0.0, random.uniform(epsilon / self.exploration_worker_difference,
                                          epsilon * self.exploration_worker_difference))
        return epsilon

    def pick_action_and_log_prob(self, policy, state, epsilon_exploration=None, forward=None):
        """使用策略选择一个动作"""
        state = torch.from_numpy(state).float().unsqueeze(0)  # 状态转为tensor类型
        if forward == "task":
            actor_output = policy.forward_task(state)
            action_size = self.actions_size[0]
        elif forward == "machine":
            actor_output = policy.forward_machine(state)
            action_size = self.actions_size[1]
        else:
            raise MyError("采集动作报错：未定义该网络")
        # 动作分布实例
        action_distribution = create_actor_distribution(self.action_types, actor_output, action_size)  # 动作分布实例
        action = action_distribution.sample().cpu().numpy()  # 采样一个动作
        if self.action_types == "DISCRETE":
            if random.random() <= epsilon_exploration:
                action = random.randint(0, action_size - 1)
            else:
                action = action[0]
        action_log_prob = self.calculate_log_action_probability(action, action_distribution)
        return action, action_log_prob

    def get_critic_value(self, policy, state):
        """返回评论家网络值"""
        state = torch.from_numpy(state).float().unsqueeze(0)
        critic_output = policy.forward_critic(state)
        return critic_output

    def calculate_log_action_probability(self, actions, action_distribution):
        """计算所选动作的log概率"""
        policy_distribution_log_prob = action_distribution.log_prob(torch.Tensor([actions]))
        return policy_distribution_log_prob

    def calculate_total_loss(self):
        """计算策略网络损失和评论家损失之和"""
        discounted_returns = self.calculate_discounted_returns()
        if self.normalise_rewards:
            discounted_returns = self.normalise_discounted_returns(discounted_returns)
        critic_loss, advantages = self.calculate_critic_loss_and_advantages(discounted_returns)  # 计算评论家损失和优势函数
        actor_task_loss = self.calculate_actor_loss(advantages, self.episode_log_action_task_probabilities)
        actor_machine_loss = self.calculate_actor_loss(advantages, self.episode_log_action_machine_probabilities)
        total_loss = critic_loss + actor_machine_loss + actor_task_loss
        # print("总损失：", total_loss)
        return total_loss

    def calculate_discounted_returns(self):
        """
        计算一集的累计折现收益，然后我们将在学习迭代中使用：蒙特卡洛估计计算V(s)_target 值
        """
        discounted_returns = [0]
        for ix in range(len(self.episode_states)):
            return_value = self.episode_rewards[-(ix + 1)] + self.discount_rate*discounted_returns[-1]
            discounted_returns.append(return_value)
        discounted_returns = discounted_returns[1:]
        discounted_returns = discounted_returns[::-1]
        return discounted_returns

    def normalise_discounted_returns(self, discounted_returns):
        """通过除以该时段收益的均值和标准差，使贴现收益归一化"""
        mean = np.mean(discounted_returns)
        std = np.std(discounted_returns)
        discounted_returns -= mean
        discounted_returns /= (std + 1e-5)
        return discounted_returns

    def calculate_critic_loss_and_advantages(self, all_discounted_returns):
        """计算评论家的损失和优势"""
        critic_values = torch.cat(self.critic_outputs)
        advantages = torch.Tensor(all_discounted_returns) - critic_values  # 计算优势函数值|V(s_t)_target - V(s_t)_critic
        advantages = advantages.detach()
        critic_loss = (torch.Tensor(all_discounted_returns) - critic_values)**2
        critic_loss = critic_loss.mean()
        return critic_loss, advantages

    def calculate_actor_loss(self, advantages, episode_log_action_probabilities):
        """计算参与者的损失"""
        action_log_probabilities_for_all_episodes = torch.cat(episode_log_action_probabilities)
        actor_loss = -1.0 * action_log_probabilities_for_all_episodes * advantages
        actor_loss = actor_loss.mean()
        return actor_loss

    def put_gradients_in_queue(self, total_loss):
        """将梯度放入队列中，以供优化过程用于更新共享模型"""
        # 线程工序策略网络梯度加入队列
        self.local_actor_critic_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_actor_critic_model.parameters(), self.gradient_clipping_norm)  # 梯度裁剪
        gradients = [param.grad.clone() for param in self.local_actor_critic_model.parameters()]  # 线程子网络梯度
        self.gradient_updates_queue.put(gradients)

# 测试算法
if __name__ == '__main__':
    da3c_object = DA3C()
    da3c_object.run_n_episodes()
