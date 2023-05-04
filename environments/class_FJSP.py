"""
������ҵ������Ȼ����ඨ��
"""
import copy
import random, math
import time
import numpy as np
from SO_DFJSP_instance import Instance
from data.data_process import Data
import matlab
import matlab.engine
from scipy.optimize import minimize, NonlinearConstraint

class Order():
    """��������"""
    def __init__(self, s, arrive, delivery, count_kind):
        # ��������
        self.order_node = s
        self.time_arrive = arrive  # ����ʱ��
        self.time_delivery = delivery  # ����ʱ��
        self.count_kind = count_kind  # �����ĸ��ֹ���������

class Kind():
    """����������"""
    def __init__(self, r):
        self.kind = r
        self.job_arrive_list = []  # �Ѿ�����Ĺ��������б�
        self.job_unfinished_list = []  # δ�ӹ���ɵĹ��������б�
    @property
    def job_number(self):
        """�����͹����ѵ��﹤����"""
        return len(self.job_arrive_list)

class Tasks(Kind):
    """���幤��o_rj��"""
    def __init__(self, r, j):
        Kind.__init__(self, r)  # ���ø���Ĺ���
        # ��������
        self.task = j  # ��������
        self.machine_tuple = None  # ��ѡ�ӹ��������Ԫ��
        # ��������
        self.job_now_list = []  # ���ڸù���εĹ��������б�
        self.job_unprocessed_list = []  # �ù����δ���ӹ��Ĺ��������б�
        self.task_unprocessed_list = []  # �ù���λ�δ�ӹ��Ĺ�������б�
        self.task_processed_list = []  # �ù�����Ѽӹ��Ĺ�������б�
        # �����������
        self.fluid_time = None  # ����ģ���иù���ļӹ�ʱ��
        self.fluid_rate = None  # ����ģ���мӹ��ù��������
        self.fluid_number = None  # ���ڸù���ε���������
        self.fluid_unprocessed_number = None  # δ���ӹ���������
        self.fluid_unprocessed_number_start = None  # ��������ʱ��δ���ӹ���������
        # ��������������
        self.fluid_machine_list = []  # ����ģ���п�ѡ�ӹ���������б�

    # ��������
    @property
    def gap(self):
        """����gap_rjֵ"""
        return (len(self.task_unprocessed_list) - self.fluid_unprocessed_number)/self.fluid_unprocessed_number_start
    @property
    def finish_rate(self):
        """o_rj�����"""
        return len(self.task_processed_list)/(len(self.task_unprocessed_list) + len(self.task_processed_list))
    @property
    def due_date_min(self):
        return self.job_now_list[0].due_data
    @property
    def due_date_ave(self):
        return sum([job.due_data for job in self.job_unprocessed_list])/len(self.job_unprocessed_list)

class Job(Kind):
    """������"""
    def __init__(self, r, n):
        Kind.__init__(self, r)  # ���ø���Ĺ���
        # ��������
        self.number = n  # �ù������͵ĵڼ�������
        # ��������
        self.due_date = None  # �ù����Ľ���
        self.task_list = []  # �Ѵ���������б�

class Task(Tasks, Job):
    """������"""
    def __init__(self, r, n, j):
        Tasks.__init__(self, r, j)  # ���ø���Ĺ���
        Job.__init__(self, r, n)  # ���ø��๹��
        # ��������
        self.machine = None  # ѡ��Ļ���
        self.time_end = None  # �ӹ�����ʱ��
        self.time_begin = None  # �ӹ���ʼʱ��
        self.time_cost = None  # �ӹ�ʱ��

class Machine():
    """������"""
    def __init__(self, m):
        # ��������
        self.machine_node = m  # �������
        self.kind_task_tuple = None  # ��ѡ�ӹ���������Ԫ��
        self.process_rate_rj_dict = {}  # �ӹ����������͵�����
        # ��������
        self.machine_state = 0  # ����״̬
        self.time_end = 0  # �����깤ʱ��
        self.task_list = []  # �����Ѽӹ���������б�
        self.unprocessed_rj_dict = {}  # δ��m�ӹ��ĸ��������͵Ĺ�������
        # ���帽������
        self.fluid_kind_task_list = []  # ������п�ѡ�ӹ����������б�
        self.time_rate_rj_dict = {}  # ������з�������������͵�ʱ�����
        self.fluid_process_rate_rj_dict = {}  # ������мӹ����������͵�����
        self.gap_rj_dict = {}  # ����gap_mrjֵ rj
        self.fluid_unprocessed_rj_dict = {}  # δ������m�ӹ��ĸ�������������
        self.fluid_unprocessed_rj_arrival_dict = {}  # ��������ʱ��δ��m�ӹ��ĸ�������������

    def utilize_rate(self, step_time):
        """������"""
        return sum([task.time_cost for task in self.task_list])/max(step_time, self.time_end)

# ����ʵ����
class FJSP(Instance):
    """������ҵ���������"""
    def __init__(self, DDT, M, S):
        Instance.__init__(self, DDT, M, S)
        # ��װ��������
        self.step_count = 0  # ���߲�
        self.step_time = 0  # ʱ���
        self.last_observation_state = None  # ��һ���۲쵽��״̬ v(t-1)
        self.observation_state = None  # ��ǰʱ�䲽��״̬ v(t)
        self.state_gap = None  # v(t) - v(t-1)
        self.state = None  # s(t)
        self.next_state = None  # ��һ��״̬  s(t+1)
        self.reward = None  # ��ʱ����
        self.done = False  # �Ƿ�Ϊ��ֹ״̬

        # ʵ�����������͡��������������͡�����ͻ��������ֵ�
        self.task_kind_dict = {(r, j): Tasks(r, j) for r in self.kind_tuple for j in self.task_r_dict[r]}  # �������Ͷ����ֵ�
        self.order_dict = {s: Order(s, self.time_arrive_s_dict[s], self.time_delivery_s_dict[s], self.count_sr_dict[s])
                           for s in self.order_tuple}  # ���󶩵��ֵ�
        self.kind_dict = {r: Kind(r) for r in self.kind_tuple}  # �������Ͷ����ֵ�
        self.machine_dict = {m: Machine(m) for m in self.machine_tuple}  # ���������ֵ�
        self.task_kind_number_dict = {}  # (r,n,j) ��������ֵ� �����������
        self.job_dict = {}  # (r,n)  # ���������ֵ�
        self.reset_parameter()  # ��ʼ�����������е��б���ֵ�
        # ����ź͹������ͺŻ�������
        self.fluid_tuple = tuple(fluid for fluid in range(len(self.kind_task_tuple)))  # ������
        self.kind_task_fluid_dict = {fluid: self.kind_task_tuple[fluid] for fluid in self.fluid_tuple}  # �����Ӧ�Ĺ�������
        self.fluid_kind_task_dict = {self.kind_task_tuple[fluid]: fluid for fluid in self.fluid_tuple}  # �������Ͷ�Ӧ������
        self.process_rate_m_rj_list = [[self.machine_dict[m].process_rate_rj_dict[(r, j)] if (r, j) in self.kind_task_m_dict[m] else 0
                                        for (r, j) in self.kind_task_tuple] for m in self.machine_tuple]  # �����ӹ���������
        self.task_number_r_list = [len(self.task_r_dict[r]) for r in self.kind_tuple]  # ���������͵Ĺ�����
        self.fluid_end_tuple = tuple(self.fluid_kind_task_dict[(r, self.task_r_dict[r][-1])] for r in self.kind_tuple)
        # ��ʼ������������# �¶����������¸��ֵ����
        self.reset_object_add()
        # ��ʼ�����л����б�Ϳ�ѡ���������б�
        self.machine_idle = []  # ���л�������б�
        self.kind_task_list = []  # ��ѡ�������ͱ���б�


    def reset_parameter(self):
        """��ʼ�����ֵ�Ͳ���"""
        for r, kind in self.kind_dict.items():
            kind.job_arrive_list = []  # �Ѿ�����Ĺ��������б�
            kind.job_unfinished_list = []  # δ�ӹ���ɵĹ��������б�
        for (r, j), task_kind_object in self.task_kind_dict.items():
            task_kind_object.machine_tuple = self.machine_rj_dict[(r, j)]  # ��ѡ�ӹ��������Ԫ��
            task_kind_object.job_now_list = []  # ���ڸù���εĹ��������б�
            task_kind_object.job_unprocessed_list = []  # �ù����δ���ӹ��Ĺ��������б�
            task_kind_object.task_unprocessed_list = []  # �ù���λ�δ�ӹ��Ĺ�������б�
            task_kind_object.task_processed_list = []  # �ù�����Ѽӹ��Ĺ�������б�
            task_kind_object.fluid_machine_list = []  # ����ģ���п�ѡ�ӹ�����
        for m, machine_object in self.machine_dict.items():
            machine_object.kind_task_tuple = self.kind_task_m_dict[m]  # ��ѡ�ӹ���������Ԫ��
            machine_object.machine_state = 0  # ����״̬
            machine_object.time_end = 0  # �����깤ʱ��
            machine_object.task_list = []  # �����Ѽӹ���������б�
            machine_object.process_rate_rj_dict = {(r, j): 1/self.time_mrj_dict[m][(r, j)]
                                                   for (r, j) in self.kind_task_m_dict[m]}  # �ӹ����������͵�����
            machine_object.unprocessed_rj_dict = {}  # δ��m�ӹ��Ĺ���o_rj������ (r,j)
            # ���帽������
            machine_object.fluid_kind_task_list = []  # ������п�ѡ�ӹ����������б�
            machine_object.time_rate_rj_dict = {}  # ������з�������������͵�ʱ�����
            machine_object.fluid_process_rate_rj_dict = {}  # ������мӹ����������͵�����
            machine_object.gap_rj_dict = {}  # ����gap_mrjֵ rj
            machine_object.fluid_unprocessed_rj_dict = {}  # �������δ������m�ӹ��ĸ�������������
            machine_object.fluid_unprocessed_rj_arrival_dict = {}  # ��������ʱ��δ��m�ӹ��ĸ�������������

        return None

    def reset_object_add(self):
        """����ֵ����"""
        order_object = self.order_dict[0]  # ������¶���
        # ���¹��������ֵ䡢�������Ͷ����ֵ䡢��������ֵ䡢���������ֵ䡢
        for r in self.kind_tuple:
            n_start = len(self.kind_dict[r].job_arrive_list)
            n_end = n_start + order_object.count_kind[r]
            for n in range(n_start, n_end):
                job_object = Job(r, n)  # ��������
                job_object.due_date = order_object.time_delivery  # ��������
                job_object.task_list = []
                self.kind_dict[r].job_arrive_list.append(job_object)
                self.kind_dict[r].job_unfinished_list.append(job_object)
                self.job_dict[(r, n)] = job_object  # ���빤���ֵ�
                self.task_kind_dict[(r, 0)].job_now_list.append(job_object)
                for j in self.task_r_dict[r]:
                    task_object = Task(r, n, j)  # �������
                    task_object.due_date = self.job_dict[(r, n)].due_date  # ������
                    self.task_kind_dict[(r, j)].job_unprocessed_list.append(job_object)
                    self.task_kind_dict[(r, j)].task_unprocessed_list.append(task_object)
                    self.task_kind_number_dict[(r, n, j)] = task_object  # ���빤���ֵ�
        # ��ʼ����������
        for (r, j), task_kind_object in self.task_kind_dict.items():
            task_kind_object.fluid_number = len(task_kind_object.job_now_list)  # ���ڸù���ε���������
            task_kind_object.fluid_unprocessed_number = len(task_kind_object.task_unprocessed_list)  # δ���ӹ���������
            task_kind_object.fluid_unprocessed_number_start = len(task_kind_object.task_unprocessed_list)  # ��������ʱ��δ���ӹ�����������
        # �������ģ�͸�������ģ������
        self.fluid_model_delivery()

    def fluid_model_delivery(self):
        """
        ��������ģ�͡���������������matlab����
        ���룺����������δ�ӹ��������������ڸù������ͽ׶εĹ�����
        �������������������������͵�ʱ���������������ӳ�ʱ�䡢���������Ϳ�ѡ�ӹ���������������ѡ��������+�ӹ����ʡ����������ͼӹ�����+ʱ��
        """
        # ���ɸ������������
        fluid_number_list = [self.task_kind_dict[self.kind_task_fluid_dict[fluid]].fluid_unprocessed_number_start for fluid in self.fluid_tuple]
        fluid_number_now_list = [self.task_kind_dict[self.kind_task_fluid_dict[fluid]].fluid_number for fluid in self.fluid_tuple]
        # ����ÿ�ֹ�����ĩβ����ÿ������Ľ���ʱ��
        fluid_end_number_list = [fluid_number_list[fluid] for fluid in self.fluid_end_tuple]  # �������������һ������Ĵ��ӹ�����
        fluid_end_number_max = max(fluid_end_number_list)  # ���ֵ
        # ��ʼ���������������һ���������͵ĸ������Ľ���ʱ��
        fluid_end_number_delivery = np.zeros([len(fluid_end_number_list), fluid_end_number_max])
        # ����ÿ�������Ľ���
        for f, fluid in enumerate(self.fluid_end_tuple):
            rj_fluid = self.kind_task_fluid_dict[fluid]
            for n, task_object in enumerate(self.task_kind_dict[rj_fluid].task_unprocessed_list):
                fluid_end_number_delivery[f, n] = task_object.due_date - self.step_time
        # ����matlab�������ģ��
        time_start = time.clock()  # matlab������⿪ʼʱ��
        engine = matlab.engine.start_matlab()  # ��������
        fluid_solve = engine.line_cplex(matlab.double(self.machine_tuple), matlab.double(self.fluid_tuple),
                                        matlab.double(self.process_rate_m_rj_list), matlab.double(fluid_number_list),
                                        matlab.double(fluid_number_now_list), matlab.double(self.task_number_r_list),
                                        matlab.double(fluid_end_number_delivery), nargout=2)
        engine.exit()  # �ر�����
        print("matlab����ʱ��", time.clock() - time_start)

        return fluid_solve[0], fluid_solve[1]

# ���Ի���
if __name__ == '__main__':
    DDT = 1.0
    M = 15
    S = 4
    fjsp_object = FJSP(DDT, M, S)
