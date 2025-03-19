# type: ignore[all]

"""
手动实现scalar类（简单的计算图节点）和反向传播算法，以及相应的可视化工具
"""

import math
from graphviz import Digraph


class Scalar:

    def __init__(self, value, prevs=[], op=None, label="", requires_grad=True):
        self.value = value
        self.prevs = prevs
        self.op = op
        self.label = label
        self.requires_grad = requires_grad

        # 节点的全局偏导数
        self.grad = 0.0
        # 节点的局部偏导数（前续节点的偏导数）
        self.grad_wrt = {}
        # 作图需要，实际上对计算没有作用
        self.back_prop = {}

    def __repr__(self):
        return f"Scalar(value={self.value:.2f}, grad={self.grad:.2f}, label={self.label}), op={self.op}"

    def __add__(self, other):
        if not isinstance(other, Scalar):
            other = Scalar(other, requires_grad=False)
        out = Scalar(self.value + other.value, [self, other], "+")
        out.requires_grad = self.requires_grad or other.requires_grad
        out.grad_wrt = {self: 1, other: 1}
        return out

    def __sub__(self, other):
        if not isinstance(other, Scalar):
            other = Scalar(other, requires_grad=False)
        out = Scalar(self.value - other.value, [self, other], "-")
        out.requires_grad = self.requires_grad or other.requires_grad
        out.grad_wrt = {self: 1, other: -1}
        return out

    def __mul__(self, other):
        if not isinstance(other, Scalar):
            other = Scalar(other, requires_grad=False)
        out = Scalar(self.value * other.value, [self, other], "*")
        out.requires_grad = self.requires_grad or other.requires_grad
        out.grad_wrt = {self: other.value, other: self.value}
        return out

    def __pow__(self, other):
        """
        只考虑对底数求导，而指数被限制为常数，常数求导为0，且常数无需进行反向传播。
        """
        assert isinstance(other, (int, float)), "only support int or float"
        out = Scalar(self.value**other, [self], f"**{other}")
        out.requires_grad = self.requires_grad
        out.grad_wrt = {self: other * self.value ** (other - 1)}
        return out

    def __rsub__(self, other):
        if not isinstance(other, Scalar):
            other = Scalar(other, requires_grad=False)
        out = Scalar(other.value - self.value, [other, self], "-")
        out.requires_grad = other.requires_grad or self.requires_grad
        out.grad_wrt = {other: 1, self: -1}
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.value))
        out = Scalar(s, [self], "sigmoid")
        out.requires_grad = self.requires_grad
        out.grad_wrt = {self: s * (1 - s)}
        return out

    def backward(self, fn=None):
        """
        反向传播，计算梯度。
        由当前节点出发，求解以当前节点为顶点的计算图中每个节点的偏导数。
        fn：画图函数。
        """

        def _topo_sort(node):
            def _dfs(node):
                if node not in visited:
                    visited.add(node)
                    for prev in node.prevs:
                        _dfs(prev)
                    order.append(node)

            order, visited = [], set()
            _dfs(node)
            return order

        def _compute_grad_wrt(node, cg_grad):
            # 作图需要，对计算无用
            node.back_prop = {}
            # 存储节点的梯度累计（可能来自多次反向传播，批量训练的基础）
            node.grad += cg_grad[node]
            for prev in node.prevs:
                grad_spread = cg_grad[node] * node.grad_wrt[prev]
                cg_grad[prev] = cg_grad.get(prev, 0.0) + grad_spread
                node.back_prop[prev] = node.back_prop.get(prev, 0.0) + grad_spread

        order = _topo_sort(self)
        re = []
        # 保存当前计算图中节点的梯度累计
        # 由于是反向传播算法起点，故 ∂self/∂self = 1
        # 其余节点的梯度初始化为 0
        # cg_grad 只跟踪当前这一次反向传播的梯度值
        cg_grad = {self: 1}
        for node in reversed(order):
            _compute_grad_wrt(node, cg_grad)
            if fn:
                re.append(fn(self, "backward"))
        return re


def _get_node_attr(node, direction="forward"):
    """
    节点的属性
    """
    node_type = _get_node_type(node)
    # 设置字体
    res = {"fontname": "Menlo"}

    def _forward_attr():
        if node_type == "param":
            node_text = f"{{ grad=None | value={node.value:.2f} | {node.label}}}"
            res.update(
                dict(
                    label=node_text,
                    shape="record",
                    fontsize="10",
                    fillcolor="lightgreen",
                    style="filled, bold",
                )
            )
            return res
        elif node_type == "computation":
            node_text = f"{{ grad=None | value={node.value:.2f} | {node.op}}}"
            res.update(
                dict(
                    label=node_text,
                    shape="record",
                    fontsize="10",
                    fillcolor="gray94",
                    style="filled, rounded",
                )
            )
            return res
        elif node_type == "input":
            if node.label == "":
                node_text = f"input={node.value:.2f}"
            else:
                node_text = f"{node.label}={node.value:.2f}"
            res.update(dict(label=node_text, shape="oval", fontsize="10"))
            return res

    def _backward_attr():
        attr = _forward_attr()
        attr["label"] = attr["label"].replace("grad=None", f"grad={node.grad:.2f}")
        if not node.requires_grad:
            attr["style"] = "dashed"
        # 为了作图美观
        # 如果向后扩散（反向传播）的梯度等于0，或者扩散给不需要梯度的节点，那么该节点用虚线表示
        grad_back = [v if k.requires_grad else 0 for (k, v) in node.back_prop.items()]
        if len(grad_back) > 0 and sum(grad_back) == 0:
            attr["style"] = "dashed"
        return attr

    if direction == "forward":
        return _forward_attr()
    else:
        return _backward_attr()


def _get_node_type(node):
    """
    决定节点的类型，计算节点、参数以及输入数据
    """
    if node.op is not None:
        return "computation"
    if node.requires_grad:
        return "param"
    return "input"


def _trace(root):
    """
    遍历图中的所有点和边
    """
    nodes, edges = set(), set()

    def _build(v):
        if v not in nodes:
            nodes.add(v)
            for prev in v.prevs:
                edges.add((prev, v))
                _build(prev)

    _build(root)
    return nodes, edges


def _draw_node(graph, node, direction="forward"):
    """
    画节点
    """
    node_attr = _get_node_attr(node, direction)
    uid = str(id(node)) + direction
    graph.node(name=uid, **node_attr)


def _draw_edge(graph, n1, n2, direction="forward"):
    """
    画边
    """
    uid1 = str(id(n1)) + direction
    uid2 = str(id(n2)) + direction

    def _draw_back_edge():
        if n1.requires_grad and n2.requires_grad:
            grad = n2.back_prop.get(n1, None)
            if grad is None:
                graph.edge(uid2, uid1, arrowhead="none", color="deepskyblue")
            elif grad == 0:
                graph.edge(
                    uid2,
                    uid1,
                    style="dashed",
                    label=f"{grad:.2f}",
                    color="deepskyblue",
                    fontname="Menlo",
                )
            else:
                graph.edge(
                    uid2,
                    uid1,
                    label=f"{grad:.2f}",
                    color="deepskyblue",
                    fontname="Menlo",
                )
        else:
            graph.edge(
                uid2, uid1, style="dashed", arrowhead="none", color="deepskyblue"
            )

    if direction == "forward":
        graph.edge(uid1, uid2)
    elif direction == "backward":
        _draw_back_edge()
    else:
        _draw_back_edge()
        graph.edge(uid1, uid2)


def draw_graph(root, direction="forward"):
    """
    图形化展示由root为顶点的计算图
    参数
    ----
    root ：Scalar，计算图的顶点
    direction ：str，向前传播（forward）或者反向传播（backward）
    返回
    ----
    re ：Digraph，计算图
    """
    nodes, edges = _trace(root)
    rankdir = "BT" if direction == "forward" else "TB"
    graph = Digraph(format="png", graph_attr={"rankdir": rankdir})
    for item in nodes:
        _draw_node(graph, item, direction)
    for n1, n2 in edges:
        _draw_edge(graph, n1, n2, direction)
    return graph
