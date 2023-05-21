"""
继承环境，并重写调度规则
知名调度规则：
FIFO:最先到达的工件-First in First out
EDD:最小交期时间工件-Earliest due date
MRT:最多剩余处理时间-Most remaining processing time
SPT:下道工序加工时间最短的工件-Shortest processing time
LPT:下道工序加工时间最长的工件-Longest processing time
CR:最小的紧迫系数-Critical ratio
random:基于工序规则和机器规则的任意选择策略
EFT:最早完成时间工件-Earliest Finish Time
复合调度规则：
CR_EFT
FIFO_EFT
MRT_SPT
MRT_FIFO
"""