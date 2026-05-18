# import torch
# import torch.nn as nn

# from torch_geometric.nn import MessagePassing


# class RelationGate(nn.Module):
#     def __init__(self, emb_size, hidden_size):
#         super().__init__()
#         self.gate_mlp = nn.Sequential(
#             nn.Linear(emb_size * 2, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, q_emb, relation_embs):
#         # q_emb: [1, emb_size]
#         # relation_embs: [num_relations, emb_size]


#         q_emb_expanded = q_emb.expand(relation_embs.size(0), -1)


#         gate_input = torch.cat([q_emb_expanded, relation_embs], dim=1)


#         return self.gate_mlp(gate_input)

# class PEConv(MessagePassing):
#     def __init__(self):
    
#         super().__init__(aggr='mean')

#     def forward(self, edge_index, x, gates):
    
#         return self.propagate(edge_index, x=x, gate=gates)

#     def message(self, x_j, gate):
#         return gate.view(-1, 1) * x_j

# class DDE(nn.Module):
#     def __init__(
#         self,
#         num_rounds,
#         num_reverse_rounds
#     ):
#         super().__init__()
        
#         self.layers = nn.ModuleList()
#         for _ in range(num_rounds):
#             self.layers.append(PEConv())
        
#         self.reverse_layers = nn.ModuleList()
#         for _ in range(num_reverse_rounds):
#             self.reverse_layers.append(PEConv())

#     def forward(
#             self,
#             topic_entity_one_hot,
#             edge_index,
#             reverse_edge_index,
#             edge_gates  
#     ):
#         result_list = []

#         h_pe = topic_entity_one_hot
#         for layer in self.layers:
#             h_pe = layer(edge_index, h_pe, edge_gates)
#             result_list.append(h_pe)

#         h_pe_rev = topic_entity_one_hot
#         for layer in self.reverse_layers:
#             h_pe_rev = layer(reverse_edge_index, h_pe_rev, edge_gates)
#             result_list.append(h_pe_rev)

#         return result_list

# class Retriever(nn.Module):
#     def __init__(
#         self,
#         emb_size,
#         topic_pe,
#         DDE_kwargs,
#         gate_kwargs
#     ):
#         super().__init__()
        
#         self.non_text_entity_emb = nn.Embedding(1, emb_size)
#         self.topic_pe = topic_pe
#         self.dde = DDE(**DDE_kwargs)
#         self.relation_gate = RelationGate(emb_size, **gate_kwargs)
        
#         pred_in_size = 4 * emb_size
#         if topic_pe:
#             pred_in_size += 2 * 2
#         pred_in_size += 2 * 2 * (DDE_kwargs['num_rounds'] + DDE_kwargs['num_reverse_rounds'])

#         self.pred = nn.Sequential(
#             nn.Linear(pred_in_size, emb_size),
#             nn.ReLU(),
#             nn.Linear(emb_size, 1)
#         )

#     def forward(
#         self,
#         h_id_tensor,
#         r_id_tensor,
#         t_id_tensor,
#         q_emb,
#         entity_embs,
#         num_non_text_entities,
#         relation_embs,
#         topic_entity_one_hot
#     ):
#         device = entity_embs.device
#         all_relation_gates = self.relation_gate(q_emb, relation_embs)
#         edge_gates = all_relation_gates[r_id_tensor]
        
#         h_e = torch.cat(
#             [
#                 entity_embs,
#                 self.non_text_entity_emb(
#                     torch.LongTensor([0]).to(device)).expand(num_non_text_entities, -1)
#             ]
#         , dim=0)
#         h_e_list = [h_e]
#         if self.topic_pe:
#             h_e_list.append(topic_entity_one_hot)

#         edge_index = torch.stack([
#             h_id_tensor,
#             t_id_tensor
#         ], dim=0)
#         reverse_edge_index = torch.stack([
#             t_id_tensor,
#             h_id_tensor
#         ], dim=0)
#         dde_list = self.dde(topic_entity_one_hot, edge_index, reverse_edge_index, edge_gates)
#         h_e_list.extend(dde_list)
#         h_e = torch.cat(h_e_list, dim=1)

#         h_q = q_emb
#         h_r = relation_embs[r_id_tensor]

#         h_triple = torch.cat([
#             h_q.expand(len(h_r), -1),
#             h_e[h_id_tensor],
#             h_r,
#             h_e[t_id_tensor]
#         ], dim=1)
        
#         return self.pred(h_triple)
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class RelationGate(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(emb_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, q_emb, relation_embs):
        q_emb_expanded = q_emb.expand(relation_embs.size(0), -1)
        gate_input = torch.cat([q_emb_expanded, relation_embs], dim=1)
        return self.gate_mlp(gate_input)

class PEConv(MessagePassing):
    # NEW: 添加 aggr_method 参数，用于支持消融实验中的 'sum' 聚合
    def __init__(self, aggr_method='mean'):
        super().__init__(aggr=aggr_method)

    def forward(self, edge_index, x, gates):
        return self.propagate(edge_index, x=x, gate=gates)

    def message(self, x_j, gate):
        return gate.view(-1, 1) * x_j

class DDE(nn.Module):
    # NEW: 接收 aggr_method 并传递给 PEConv
    def __init__(
        self,
        num_rounds,
        num_reverse_rounds,
        aggr_method='mean'
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_rounds):
            self.layers.append(PEConv(aggr_method=aggr_method))
        
        self.reverse_layers = nn.ModuleList()
        for _ in range(num_reverse_rounds):
            self.reverse_layers.append(PEConv(aggr_method=aggr_method))

    def forward(
            self,
            topic_entity_one_hot,
            edge_index,
            reverse_edge_index,
            edge_gates  
    ):
        result_list = []
        h_pe = topic_entity_one_hot
        for layer in self.layers:
            h_pe = layer(edge_index, h_pe, edge_gates)
            result_list.append(h_pe)

        h_pe_rev = topic_entity_one_hot
        for layer in self.reverse_layers:
            h_pe_rev = layer(reverse_edge_index, h_pe_rev, edge_gates)
            result_list.append(h_pe_rev)

        return result_list

class Retriever(nn.Module):
    def __init__(
        self,
        emb_size,
        topic_pe,
        DDE_kwargs,
        gate_kwargs,
        # NEW: 增加消融实验控制参数
        ablation_no_gate=False,
        ablation_sum_aggr=False,
        ablation_no_structure=False
    ):
        super().__init__()
        
        self.ablation_no_gate = ablation_no_gate
        self.ablation_no_structure = ablation_no_structure
        
        # 针对 SUM Aggregation 消融
        aggr_method = 'sum' if ablation_sum_aggr else 'mean'
        DDE_kwargs['aggr_method'] = aggr_method

        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        self.dde = DDE(**DDE_kwargs)
        self.relation_gate = RelationGate(emb_size, **gate_kwargs)
        
        # 动态计算预测层输入维度
        pred_in_size = 4 * emb_size
        if topic_pe:
            pred_in_size += 2 * 2
        
        # 如果不消融结构特征，才加上结构特征的维度
        if not self.ablation_no_structure:
            pred_in_size += 2 * 2 * (DDE_kwargs['num_rounds'] + DDE_kwargs['num_reverse_rounds'])

        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )

    def forward(
        self,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot
    ):
        device = entity_embs.device
        all_relation_gates = self.relation_gate(q_emb, relation_embs)
        
        # ABLATION 1: w/o Relation Gating (将门控值全部固定为 1)
        if self.ablation_no_gate:
            edge_gates = torch.ones_like(all_relation_gates[r_id_tensor])
        else:
            edge_gates = all_relation_gates[r_id_tensor]
        
        h_e = torch.cat(
            [
                entity_embs,
                self.non_text_entity_emb(
                    torch.LongTensor([0]).to(device)).expand(num_non_text_entities, -1)
            ]
        , dim=0)
        
        h_e_list = [h_e]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)

        # ABLATION 3: w/o Structural Features (跳过 DDE，不拼接多跳状态)
        if not self.ablation_no_structure:
            edge_index = torch.stack([h_id_tensor, t_id_tensor], dim=0)
            reverse_edge_index = torch.stack([t_id_tensor, h_id_tensor], dim=0)
            dde_list = self.dde(topic_entity_one_hot, edge_index, reverse_edge_index, edge_gates)
            h_e_list.extend(dde_list)
            
        h_e = torch.cat(h_e_list, dim=1)

        h_q = q_emb
        h_r = relation_embs[r_id_tensor]

        h_triple = torch.cat([
            h_q.expand(len(h_r), -1),
            h_e[h_id_tensor],
            h_r,
            h_e[t_id_tensor]
        ], dim=1)
        
        return self.pred(h_triple)