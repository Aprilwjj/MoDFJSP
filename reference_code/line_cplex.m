function [H1, H2] = line_cplex(machine_list, class_list, rate_machine_class_index, class_number, class_number_time, number_task)
% ����ģ����⺯��

m_number = size(machine_list,2);
k_number = size(class_list, 2);
%%% ������߱��� %%%
x = sdpvar(m_number, k_number, 'full');
%%% Ŀ�꺯�� %%%
class_finish = [];
for k = 1:k_number
    class_finish = [class_finish, class_number(k)/sum(x(:,k).*rate_machine_class_index(:,k))];
end
z = max(class_finish);
%%% ���Լ�� %%%
C = [];
% ���߱�����ΧԼ��
for m = 1:m_number
    for k = 1:k_number
        C = [C, 0<=x(m,k)<=1];
        if rate_machine_class_index(m,k)==0
            C = [C, x(m,k)==0];
        end
    end
end
% ����������Լ��
for m = 1:m_number
    C = [C, sum(x(m,:))<=1];
end
% ���ɷ��׹����༯��
kind_number = size(number_task, 2);  % ����������
class_first_task = ones(1, kind_number);  % �׹���������
for j = 2:kind_number
    class_first_task(j) = sum(number_task(1,1:j-1)) + 1;
end
% ����������Լ��
for k = 1:k_number
    if class_number_time(k)~=0|ismember(k, class_first_task)
        continue
    else
        C = [C, sum(x(:,k).*rate_machine_class_index(:,k))<=sum(x(:,k-1).*rate_machine_class_index(:,k-1))];
    end
end
%%% ���� %%%
ops = sdpsettings('verbose',0,'solver','cplex');
%%% ��� %%%
reuslt = optimize(C,z);
H1 = value(x);
H2 = value(z);
% �ж��Ƿ����ɹ�
if reuslt.problem == 0 % problem =0 �������ɹ�
    value(x)
    value(z)   
else
    disp('������');
end

end

