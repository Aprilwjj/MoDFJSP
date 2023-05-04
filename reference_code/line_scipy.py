"""
scipy ������Թ滮����
"""
def run_fluid_model(self):
    """
    ��������ģ��
    ���룺����������δ�ӹ��������������ڸù������ͽ׶εĹ�����
    �������������������������͵�ʱ���������������ӳ�ʱ�䡢���������Ϳ�ѡ�ӹ���������������ѡ��������+�ӹ����ʡ����������ͼӹ�����+ʱ��
    """

    # ���ɸ������������
    fluid_number = {(r, j): self.task_kind_dict[(r, j)].fluid_unprocessed_number_start for (r, j) in
                    self.kind_task_tuple}
    fluid_number_time = {(r, j): self.task_kind_dict[(r, j)].fluid_number for (r, j) in self.kind_task_tuple}
    task_end_r_dict = {r: self.task_r_dict[r][-1] for r in self.kind_tuple}
    # ���߱���
    mrj_tuple = tuple((m, (r, j)) for m in self.machine_tuple for (r, j) in self.kind_task_m_dict[m])
    x_mrj_dict = {(m, (r, j)): mrj_tuple.index((m, (r, j))) for m in self.machine_tuple for (r, j) in
                  self.kind_task_m_dict[m]}
    mrj_x_dict = {mrj_tuple.index((m, (r, j))): (m, (r, j)) for m in self.machine_tuple for (r, j) in
                  self.kind_task_m_dict[m]}
    # �������ķ�Χ
    bound = [(0, 1) for (m, (r, j)) in mrj_tuple]
    # ���Լ������
    cons_temp = []
    for m in self.machine_tuple:
        cons_temp.append({'type': 'ineq',
                          'fun': lambda x: 1 - sum(x[x_mrj_dict[(m, (r, j))]] for (r, j) in self.kind_task_m_dict[m])})
    cons = tuple(cons_temp)
    # ���ģ��
    solution = minimize(self.objective, np.ones(len(mrj_tuple)), bounds=bound, constraints=cons)
    print(solution)
    print(solution.x)
    x_mrj_result_dict = {(m, (r, j)): solution.x[x_mrj_dict[(m, (r, j))]] for m in self.machine_tuple for (r, j) in
                         self.kind_task_m_dict[m]}
    print(x_mrj_result_dict)


def objective(self, x):
    """Ŀ�꺯��"""
    # ���߱���
    mrj_tuple = tuple((m, (r, j)) for m in self.machine_tuple for (r, j) in self.kind_task_m_dict[m])
    x_mrj_dict = {(m, (r, j)): mrj_tuple.index((m, (r, j))) for m in self.machine_tuple for (r, j) in
                  self.kind_task_m_dict[m]}
    mrj_x_dict = {mrj_tuple.index((m, (r, j))): (m, (r, j)) for m in self.machine_tuple for (r, j) in
                  self.kind_task_m_dict[m]}
    # ������ز���
    fluid_number = {(r, j): self.task_kind_dict[(r, j)].fluid_unprocessed_number_start for (r, j) in
                    self.kind_task_tuple}
    fluid_number_time = {(r, j): self.task_kind_dict[(r, j)].fluid_number for (r, j) in self.kind_task_tuple}
    task_end_rj_dict = {(r, self.task_r_dict[r][-1]) for r in self.kind_tuple}
    fluid_due_date_dict = {
        (r, j): [task_object.due_date for task_object in self.task_kind_dict[(r, j)].task_unprocessed_list] for (r, j)
        in task_end_rj_dict}
    # Ŀ���е���ز���
    process_rate_sum = {(r, j): sum(
        x[x_mrj_dict[(m, (r, j))]] * self.machine_dict[m].process_rate_rj_dict[(r, j)] for m in
        self.machine_rj_dict[(r, j)]) for (r, j) in task_end_rj_dict}
    time_finish = {(r, j): fluid_number[(r, j)] / process_rate_sum[(r, j)] for (r, j) in task_end_rj_dict}
    # �����ܵ�����ʱ��
    fluid_finish_dict = {(r, j): [time_finish[(r, j)] / len(fluid_due_date_dict[(r, j)]) * (number + 1) for number in
                                  range(len(fluid_due_date_dict[(r, j)]))] for (r, j) in task_end_rj_dict}
    due_time_dict = {(r, j): [max(0, c - d) for c, d in zip(fluid_finish_dict[(r, j)], fluid_due_date_dict[(r, j)])] for
                     (r, j) in task_end_rj_dict}
    function_objection = sum(sum(due_time_dict[(r, j)]) for (r, j) in task_end_rj_dict)
    return function_objection