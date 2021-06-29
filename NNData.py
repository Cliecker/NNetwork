from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import math
import random
import collections
import copy
import json


class DataMismatchError(Exception):
    pass


class NNData:
    class Order(Enum):
        RANDOM = "Random"
        SEQUENTIAL = "Sequential"

    class Set(Enum):
        TRAIN = "Train"
        TEST = "Test"

    @staticmethod
    def percentage_limiter(percentage: float):
        if percentage < 0:
            return 0
        elif percentage > 1:
            return 1
        else:
            return percentage

    def __init__(self, features: list = None, labels: list = None, train_factor=0.9):
        if features is None:
            features = []
        if labels is None:
            labels = []
        self._labels = None
        self._features = None
        self._train_factor = NNData.percentage_limiter(train_factor)
        self._train_indices = []
        self._test_indices = []
        self._train_pool = collections.deque()
        self._test_pool = collections.deque()
        self.load_data(features, labels)
        self.split_set(self._train_factor)

    def load_data(self, features: list = None, labels: list = None):
        if features is None:
            self._features = None
            self._labels = None
            return
        elif len(features) != len(labels):
            self._features = None
            self._labels = None
            raise DataMismatchError(self)
        else:
            try:
                self._features = np.array(features, dtype=float)
                self._labels = np.array(labels, dtype=float)
            except ValueError:
                self._features = None
                self._labels = None
                raise ValueError(self)

    def split_set(self, new_train_factor=None):
        if new_train_factor is not None:
            self._train_factor = NNData.percentage_limiter(new_train_factor)
        num_examples = len(self._features)
        train_examples = math.floor(num_examples*new_train_factor)
        example_indices = random.sample(range(num_examples), num_examples)
        self._train_indices = example_indices[:train_examples]
        self._test_indices = example_indices[train_examples:]

    def prime_data(self, target_set=None, order=None):
        if target_set == NNData.Set.TRAIN:
            self._train_pool = collections.deque(self._train_indices)
        elif target_set == NNData.Set.TEST:
            self._test_pool = collections.deque(self._test_indices)
        else:
            self._train_pool = collections.deque(self._train_indices)
            self._test_pool = collections.deque(self._test_indices)

        if order == NNData.Order.RANDOM:
            random.shuffle(self._train_pool)
            random.shuffle(self._test_pool)

    def get_one_item(self, target_set=None):
        if target_set == NNData.Set.TEST:
            if self._test_pool:
                index = collections.deque.popleft(self._test_pool)
                pair = (self._features[index], self._labels[index])
                return pair
            else:
                return None
        else:
            if self._train_pool:
                index = collections.deque.popleft(self._train_pool)
                pair = (self._features[index], self._labels[index])
                return pair
            else:
                return None

    def number_of_samples(self, target_set=None):
        if target_set == NNData.Set.TEST:
            return len(self._test_pool)
        elif target_set == NNData.Set.TRAIN:
            return len(self._train_pool)
        else:
            return len(self._test_pool)+len(self._train_pool)

    def pool_is_empty(self, target_set=None):
        if target_set == NNData.Set.TEST:
            if self._test_pool:
                return False
            else:
                return True
        else:
            if self._train_pool:
                return False
            else:
                return True

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def train_indices(self):
        return self._train_indices

    @property
    def test_indices(self):
        return self._test_indices

    @property
    def train_pool(self):
        return self._train_pool

    @property
    def test_pool(self):
        return self._test_pool

    @property
    def train_factor(self):
        return self._train_factor


class LayerType(Enum):
    INPUT = "Input"
    HIDDEN = "Hidden"
    OUTPUT = "Output"


class MultiLinkNode(ABC):
    class Side(Enum):
        UPSTREAM = "Upstream"
        DOWNSTREAM = "Downstream"

    def __init__(self):
        self._reporting_nodes = {self.Side.UPSTREAM: 0, self.Side.DOWNSTREAM: 0}
        self._reference_value = {self.Side.UPSTREAM: 0, self.Side.DOWNSTREAM: 0}
        self._neighbors = {self.Side.UPSTREAM:  [], self.Side.DOWNSTREAM: []}

    def __str__(self):
        for node in self._neighbors[self.Side.UPSTREAM]:
            print(f"Node Neighbor Upstream: {id(node)}")
        for node in self._neighbors[self.Side.DOWNSTREAM]:
            print(f"Node Neighbor Downstream: {id(node)}")
        return f"Are connected to Node ID: {id(self)}"

    @abstractmethod
    def _process_new_neighbor(self, node, side):
        pass

    def reset_neighbors(self, nodes: list, side: Side):
        self._neighbors[side] = copy.copy(nodes)
        for node in self._neighbors[side]:
            self._process_new_neighbor(node, side)
        self._reference_value[side] = 2**len(nodes) - 1


class Neurode(MultiLinkNode):
    def __init__(self, node_type, learning_rate=.05):
        self._value = 0
        self._node_type = node_type
        self._learning_rate = learning_rate
        self._weights = {}
        super().__init__()

    def _process_new_neighbor(self, node, side):
        self._weights[node] = random.random()

    def _check_in(self, node, side):
        index = self._neighbors[side].index(node)
        self._reporting_nodes[side] = self._reporting_nodes[side] | 2**index
        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes[side] = 0
            return True
        else:
            return False

    def get_weight(self, node):
        return self._weights[node]

    @property
    def value(self):
        return self._value

    @property
    def node_type(self):
        return self._node_type

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value


class FFNeurode(Neurode):
    def __init__(self, my_type):
        super().__init__(my_type)

    @staticmethod
    def _sigmoid(value):
        return 1/(1+np.exp(-value))

    def _calculate_value(self):
        total_weight = 0
        for node in self._neighbors[self.Side.UPSTREAM]:
            total_weight += node.value*self.get_weight(node)
        self._value = self._sigmoid(total_weight)

    def _fire_downstream(self):
        for node in self._neighbors[self.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)

    def data_ready_upstream(self, node):
        if self._check_in(node, self.Side.UPSTREAM):
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value):
        self._value = input_value
        for node in self._neighbors[self.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)


class BPNeurode(Neurode):
    def __init__(self, my_type):
        self._delta = 0
        super().__init__(my_type)

    @staticmethod
    def _sigmoid_derivative(value):
        return value*(1-value)

    def _calculate_delta(self, expected_value=None):
        if self._node_type is LayerType.OUTPUT:
            self._delta = (expected_value-self.value)*self._sigmoid_derivative(self.value)
        else:
            total_delta_weight = 0
            for node in self._neighbors[self.Side.DOWNSTREAM]:
                total_delta_weight += node.delta*node.get_weight(self)
            self._delta = total_delta_weight*self._sigmoid_derivative(self.value)

    def data_ready_downstream(self, node):
        if self._check_in(node, self.Side.DOWNSTREAM):
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()

    def set_expected(self, expected_value):
        self._calculate_delta(expected_value)
        for node in self._neighbors[self.Side.UPSTREAM]:
            node.data_ready_downstream(self)

    def adjust_weights(self, node, adjustment):
        self._weights[node] += adjustment

    def _update_weights(self):
        for node in self._neighbors[self.Side.DOWNSTREAM]:
            adjustment = node.learning_rate*self.value*node.delta
            node.adjust_weights(self, adjustment)

    def _fire_upstream(self):
        for node in self._neighbors[self.Side.UPSTREAM]:
            node.data_ready_downstream(self)

    @property
    def delta(self):
        return self._delta


class FFBPNeurode(FFNeurode, BPNeurode):
    pass


class DoublyLinkedList:
    class EmptyListError(Exception):
        pass

    def __init__(self):
        self._head = None
        self._tail = None
        self._curr = None

    def add_to_head(self, data):
        new_node = Node(data)
        new_node.next = self._head
        self._head = new_node
        if self._head.next is not None:
            self._head.next.back = self._head
        self.reset_to_head()
        if self._tail is None:
            self._tail = self._head
        self.reset_to_head()

    def remove_from_head(self):
        if self._head is None:
            raise DoublyLinkedList.EmptyListError
        ret_val = self._head.data
        self._head.next.back = None
        self._head = self._head.next
        self.reset_to_head()
        return ret_val

    def reset_to_head(self):
        self._curr = self._head
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        else:
            return self._curr.data

    def reset_to_tail(self):
        self._curr = self._tail
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        else:
            return self._curr.data

    def move_forward(self):
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        else:
            curr_node = self._curr.next
        if curr_node is None:
            raise IndexError
        else:
            self._curr = curr_node
            return self._curr.data

    def move_back(self):
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        else:
            curr_node = self._curr.back
        if curr_node is None:
            raise IndexError
        else:
            self._curr = curr_node
            return self._curr.data

    def add_after_curr(self, data):
        if self._curr is None:
            self.add_to_head(data)
            return
        new_node = Node(data)
        new_node.next = self._curr.next
        new_node.back = self._curr
        if self._curr.next is None:
            self._curr.next = new_node
        else:
            self._curr.next.back = new_node
            self._curr.next = new_node
        curr = self._head
        while curr is not None:
            if curr.next is None:
                self._tail = curr
                return None
            curr = curr.next

    def remove_after_curr(self):
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        if self._curr.next is None:
            raise IndexError
        if self._curr.next == self._tail:
            self._tail = self._curr
            self._curr.next = None
            return None
        ret_value = self._curr.next.data
        self._curr.next = self._curr.next.next
        self._curr.next.back = self._curr
        return ret_value

    def find(self, value):
        curr_pos = self._head
        while curr_pos is not None:
            if curr_pos.data == value:
                return curr_pos.data
            curr_pos = curr_pos.next
        return None

    def delete(self, value):
        self.reset_to_head()
        if self._curr is None:
            return None
        if self._curr.data == value:
            return self.remove_from_head()
        while self._curr is not None:
            if self._curr.next.data == value:
                ret_value = self.remove_after_curr()
                self.reset_to_head()
                return ret_value
            self._curr = self._curr.next
        self.reset_to_head()
        return None

    def get_current_data(self):
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        return self._curr.data

    def __iter__(self):
        self._curr_iter = self._head
        return self

    def __next__(self):
        if self._curr_iter is None:
            raise StopIteration
        ret_val = self._curr_iter.data
        self._curr_iter = self._curr_iter.next
        return ret_val


class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.back = None


class LayerList(DoublyLinkedList):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        input_list = []
        output_list = []
        for i in range(inputs):
            input_list.append(FFBPNeurode(LayerType.INPUT))
        for i in range(outputs):
            output_list.append(FFBPNeurode(LayerType.OUTPUT))
        for neurode in input_list:
            neurode.reset_neighbors(output_list, neurode.Side.DOWNSTREAM)
        for neurode in output_list:
            neurode.reset_neighbors(input_list, neurode.Side.UPSTREAM)
        self.add_to_head(input_list)
        self.add_after_curr(output_list)

    def add_layer(self, num_nodes: int):
        if self.get_current_data() == self._tail.data:
            raise IndexError
        hidden_list = []
        for i in range(num_nodes):
            hidden_list.append(FFBPNeurode(LayerType.HIDDEN))
        for neurode in hidden_list:
            neurode.reset_neighbors(self.get_current_data(), neurode.Side.UPSTREAM)
        self.move_forward()
        for neurode in hidden_list:
            neurode.reset_neighbors(self.get_current_data(), neurode.Side.DOWNSTREAM)
        self.move_back()
        for neurode in self.get_current_data():
            neurode.reset_neighbors(hidden_list, neurode.Side.DOWNSTREAM)
        self.move_forward()
        for neurode in self.get_current_data():
            neurode.reset_neighbors(hidden_list, neurode.Side.UPSTREAM)
        self.move_back()
        self.add_after_curr(hidden_list)

    def remove_layer(self):
        self.move_forward()
        if self.get_current_data() == self._tail.data:
            self.move_back()
            raise IndexError
        self.move_back()
        self.remove_after_curr()
        upstream_nodes = self.get_current_data()
        self.move_forward()
        for neurode in self.get_current_data():
            neurode.reset_neighbors(upstream_nodes, neurode.Side.UPSTREAM)
        downstream_nodes = self.get_current_data()
        self.move_back()
        for neurode in self.get_current_data():
            neurode.reset_neighbors(downstream_nodes, neurode.Side.DOWNSTREAM)

    @property
    def input_nodes(self):
        return self.find(self._head.data)

    @property
    def output_nodes(self):
        return self.find(self._tail.data)


class FFBPNetwork:
    class EmptySetException(Exception):
        pass

    def __init__(self, num_inputs: int, num_outputs: int):
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._network = LayerList(num_inputs, num_outputs)

    def add_hidden_layer(self, num_nodes: int, position=0):
        self._network.reset_to_head()
        for i in range(position):
            self._network.move_forward()
        self._network.add_layer(num_nodes)

    def train(self, data_set: NNData, epochs=1000, verbosity=2, order=NNData.Order.RANDOM):
        if data_set is None:
            raise FFBPNetwork.EmptySetException
        rmse = 0
        for epoch in range(epochs):
            rmse = 0
            self._network.reset_to_head()
            data_set.prime_data(NNData.Set.TRAIN, order)
            num_samples = data_set.number_of_samples(NNData.Set.TRAIN)
            while not data_set.pool_is_empty(NNData.Set.TRAIN):
                pair = data_set.get_one_item()
                features = pair[0]
                labels = pair[1]
                total_error = 0
                output_neurodes = []
                for k in range(len(self._network.input_nodes)):
                    self._network.input_nodes[k].set_input(features[k])
                for i in range(len(self._network.output_nodes)):
                    output_neurodes.append(self._network.output_nodes[i].value)
                    error = self._network.output_nodes[i].value - labels[i]
                    total_error += math.pow(error, 2)
                    self._network.output_nodes[i].set_expected(labels[i])
                rmse += total_error/(len(labels)*num_samples)
                if epoch % 1000 == 0 and verbosity > 1:
                    print(f"Sample {features} expected {labels} produced {output_neurodes}")
            rmse = math.pow(rmse, 0.5)
            if epoch % 100 == 0 and verbosity > 0:
                print(f"The epoch: {epoch}  The RMSE: {rmse}")
        print(f"Final Epoch RMSE: {rmse}")

    def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL):
        if data_set is None:
            raise FFBPNetwork.EmptySetException
        rmse = 0
        self._network.reset_to_head()
        data_set.prime_data(NNData.Set.TEST, order)
        num_samples = data_set.number_of_samples(NNData.Set.TEST)
        while not data_set.pool_is_empty(NNData.Set.TEST):
            pair = data_set.get_one_item(NNData.Set.TEST)
            features = pair[0]
            labels = pair[1]
            total_error = 0
            output_neurodes = []
            for k in range(len(self._network.input_nodes)):
                self._network.input_nodes[k].set_input(features[k])
            for i in range(len(self._network.output_nodes)):
                output_neurodes.append(self._network.output_nodes[i].value)
                error = self._network.output_nodes[i].value - labels[i]
                total_error += math.pow(error, 2)
            rmse += total_error / (len(labels)*num_samples)
            print(f"Sample {features} expected {labels} produced {output_neurodes}")
        rmse = math.pow(rmse, 0.5)
        print(f"Final Test RMSE: {rmse}")


class MultiTypeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, NNData):
            return {"__NNData__": o.__dict__}
        elif isinstance(o, collections.deque):
            return {"__deque__": list(o)}
        elif isinstance(o, np.ndarray):
            return {"__NDarray__": np.ndarray.tolist(o)}
        else:
            super().default(o)


def multi_type_decoder(o):
    if "__NNData__" in o:
        item = o["__NNData__"]
        ret_obj = NNData(item["_features"], item["_labels"], item["_train_factor"])
        ret_obj._train_indices = item["_train_indices"]
        ret_obj._test_indices = item["_test_indices"]
        ret_obj._train_pool = item["_train_pool"]
        ret_obj._test_pool = item["_test_pool"]
        return ret_obj
    elif "__deque__" in o:
        return collections.deque(o["__deque__"])
    elif "__NDarray__" in o:
        return np.array(o["__NDarray__"])
    else:
        return o


def load_XOR(train_factor: float = 1):
    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]
    XOR = NNData(features, labels, train_factor)
    return XOR


def load_bias_XOR(train_factor: float = 1):
    features = [[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]]
    labels = [[0], [1], [1], [0]]
    XOR = NNData(features, labels, train_factor)
    return XOR


def unit_test():
    pass


def run_iris():
    network = FFBPNetwork(4, 3)
    network.add_hidden_layer(3)
    Iris_X = [[5.1, 3.5, 1.4, 0.2], [4.9, 3, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2],
              [5, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], [4.6, 3.4, 1.4, 0.3], [5, 3.4, 1.5, 0.2],
              [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
              [4.8, 3, 1.4, 0.1], [4.3, 3, 1.1, 0.1], [5.8, 4, 1.2, 0.2], [5.7, 4.4, 1.5, 0.4],
              [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3], [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3],
              [5.4, 3.4, 1.7, 0.2], [5.1, 3.7, 1.5, 0.4], [4.6, 3.6, 1, 0.2], [5.1, 3.3, 1.7, 0.5],
              [4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 0.2], [5, 3.4, 1.6, 0.4], [5.2, 3.5, 1.5, 0.2],
              [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4],
              [5.2, 4.1, 1.5, 0.1], [5.5, 4.2, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5, 3.2, 1.2, 0.2],
              [5.5, 3.5, 1.3, 0.2], [4.9, 3.1, 1.5, 0.1], [4.4, 3, 1.3, 0.2], [5.1, 3.4, 1.5, 0.2],
              [5, 3.5, 1.3, 0.3], [4.5, 2.3, 1.3, 0.3], [4.4, 3.2, 1.3, 0.2], [5, 3.5, 1.6, 0.6],
              [5.1, 3.8, 1.9, 0.4], [4.8, 3, 1.4, 0.3], [5.1, 3.8, 1.6, 0.2], [4.6, 3.2, 1.4, 0.2],
              [5.3, 3.7, 1.5, 0.2], [5, 3.3, 1.4, 0.2], [7, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5],
              [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3],
              [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1], [6.6, 2.9, 4.6, 1.3], [5.2, 2.7, 3.9, 1.4], [5, 2, 3.5, 1],
              [5.9, 3, 4.2, 1.5], [6, 2.2, 4, 1], [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4],
              [5.6, 3, 4.5, 1.5], [5.8, 2.7, 4.1, 1], [6.2, 2.2, 4.5, 1.5], [5.6, 2.5, 3.9, 1.1],
              [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4, 1.3], [6.3, 2.5, 4.9, 1.5], [6.1, 2.8, 4.7, 1.2],
              [6.4, 2.9, 4.3, 1.3], [6.6, 3, 4.4, 1.4], [6.8, 2.8, 4.8, 1.4], [6.7, 3, 5, 1.7], [6, 2.9, 4.5, 1.5],
              [5.7, 2.6, 3.5, 1], [5.5, 2.4, 3.8, 1.1], [5.5, 2.4, 3.7, 1], [5.8, 2.7, 3.9, 1.2],
              [6, 2.7, 5.1, 1.6], [5.4, 3, 4.5, 1.5], [6, 3.4, 4.5, 1.6], [6.7, 3.1, 4.7, 1.5],
              [6.3, 2.3, 4.4, 1.3], [5.6, 3, 4.1, 1.3], [5.5, 2.5, 4, 1.3], [5.5, 2.6, 4.4, 1.2],
              [6.1, 3, 4.6, 1.4], [5.8, 2.6, 4, 1.2], [5, 2.3, 3.3, 1], [5.6, 2.7, 4.2, 1.3], [5.7, 3, 4.2, 1.2],
              [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3], [5.1, 2.5, 3, 1.1], [5.7, 2.8, 4.1, 1.3],
              [6.3, 3.3, 6, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3, 5.9, 2.1], [6.3, 2.9, 5.6, 1.8],
              [6.5, 3, 5.8, 2.2], [7.6, 3, 6.6, 2.1], [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8],
              [6.7, 2.5, 5.8, 1.8], [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2], [6.4, 2.7, 5.3, 1.9],
              [6.8, 3, 5.5, 2.1], [5.7, 2.5, 5, 2], [5.8, 2.8, 5.1, 2.4], [6.4, 3.2, 5.3, 2.3], [6.5, 3, 5.5, 1.8],
              [7.7, 3.8, 6.7, 2.2], [7.7, 2.6, 6.9, 2.3], [6, 2.2, 5, 1.5], [6.9, 3.2, 5.7, 2.3],
              [5.6, 2.8, 4.9, 2], [7.7, 2.8, 6.7, 2], [6.3, 2.7, 4.9, 1.8], [6.7, 3.3, 5.7, 2.1],
              [7.2, 3.2, 6, 1.8], [6.2, 2.8, 4.8, 1.8], [6.1, 3, 4.9, 1.8], [6.4, 2.8, 5.6, 2.1],
              [7.2, 3, 5.8, 1.6], [7.4, 2.8, 6.1, 1.9], [7.9, 3.8, 6.4, 2], [6.4, 2.8, 5.6, 2.2],
              [6.3, 2.8, 5.1, 1.5], [6.1, 2.6, 5.6, 1.4], [7.7, 3, 6.1, 2.3], [6.3, 3.4, 5.6, 2.4],
              [6.4, 3.1, 5.5, 1.8], [6, 3, 4.8, 1.8], [6.9, 3.1, 5.4, 2.1], [6.7, 3.1, 5.6, 2.4],
              [6.9, 3.1, 5.1, 2.3], [5.8, 2.7, 5.1, 1.9], [6.8, 3.2, 5.9, 2.3], [6.7, 3.3, 5.7, 2.5],
              [6.7, 3, 5.2, 2.3], [6.3, 2.5, 5, 1.9], [6.5, 3, 5.2, 2], [6.2, 3.4, 5.4, 2.3], [5.9, 3, 5.1, 1.8]]
    Iris_Y = [[1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ]]
    data = NNData(Iris_X, Iris_Y, .7)
    network.train(data, 10001, order=NNData.Order.RANDOM)
    network.test(data)


def run_sin():
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    sin_X = [[0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07], [0.08], [0.09], [0.1], [0.11], [0.12],
             [0.13], [0.14], [0.15], [0.16], [0.17], [0.18], [0.19], [0.2], [0.21], [0.22], [0.23], [0.24], [0.25],
             [0.26], [0.27], [0.28], [0.29], [0.3], [0.31], [0.32], [0.33], [0.34], [0.35], [0.36], [0.37], [0.38],
             [0.39], [0.4], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46], [0.47], [0.48], [0.49], [0.5], [0.51],
             [0.52], [0.53], [0.54], [0.55], [0.56], [0.57], [0.58], [0.59], [0.6], [0.61], [0.62], [0.63], [0.64],
             [0.65], [0.66], [0.67], [0.68], [0.69], [0.7], [0.71], [0.72], [0.73], [0.74], [0.75], [0.76], [0.77],
             [0.78], [0.79], [0.8], [0.81], [0.82], [0.83], [0.84], [0.85], [0.86], [0.87], [0.88], [0.89], [0.9],
             [0.91], [0.92], [0.93], [0.94], [0.95], [0.96], [0.97], [0.98], [0.99], [1], [1.01], [1.02], [1.03],
             [1.04], [1.05], [1.06], [1.07], [1.08], [1.09], [1.1], [1.11], [1.12], [1.13], [1.14], [1.15], [1.16],
             [1.17], [1.18], [1.19], [1.2], [1.21], [1.22], [1.23], [1.24], [1.25], [1.26], [1.27], [1.28], [1.29],
             [1.3], [1.31], [1.32], [1.33], [1.34], [1.35], [1.36], [1.37], [1.38], [1.39], [1.4], [1.41], [1.42],
             [1.43], [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.5], [1.51], [1.52], [1.53], [1.54], [1.55],
             [1.56], [1.57]]
    sin_Y = [[0], [0.00999983333416666], [0.0199986666933331], [0.0299955002024957], [0.0399893341866342],
             [0.0499791692706783], [0.0599640064794446], [0.0699428473375328], [0.0799146939691727],
             [0.089878549198011], [0.0998334166468282], [0.109778300837175], [0.119712207288919],
             [0.129634142619695], [0.139543114644236], [0.149438132473599], [0.159318206614246],
             [0.169182349066996], [0.179029573425824], [0.188858894976501], [0.198669330795061], [0.2084598998461],
             [0.218229623080869], [0.227977523535188], [0.237702626427135], [0.247403959254523],
             [0.257080551892155], [0.266731436688831], [0.276355648564114], [0.285952225104836], [0.29552020666134],
             [0.305058636443443], [0.314566560616118], [0.324043028394868], [0.333487092140814],
             [0.342897807455451], [0.35227423327509], [0.361615431964962], [0.370920469412983], [0.380188415123161],
             [0.389418342308651], [0.398609327984423], [0.40776045305957], [0.416870802429211], [0.425939465066],
             [0.43496553411123], [0.44394810696552], [0.452886285379068], [0.461779175541483], [0.470625888171158],
             [0.479425538604203], [0.488177246882907], [0.496880137843737], [0.505533341204847],
             [0.514135991653113], [0.522687228930659], [0.531186197920883], [0.539632048733969],
             [0.548023936791874], [0.556361022912784], [0.564642473395035], [0.572867460100481],
             [0.581035160537305], [0.58914475794227], [0.597195441362392], [0.60518640573604], [0.613116851973434],
             [0.62098598703656], [0.628793024018469], [0.636537182221968], [0.644217687237691], [0.651833771021537],
             [0.659384671971473], [0.666869635003698], [0.674287911628145], [0.681638760023334],
             [0.688921445110551], [0.696135238627357], [0.70327941920041], [0.710353272417608], [0.717356090899523],
             [0.724287174370143], [0.731145829726896], [0.737931371109963], [0.744643119970859],
             [0.751280405140293], [0.757842562895277], [0.764328937025505], [0.770738878898969],
             [0.777071747526824], [0.783326909627483], [0.78950373968995], [0.795601620036366], [0.801619940883777],
             [0.807558100405114], [0.813415504789374], [0.819191568300998], [0.82488571333845], [0.83049737049197],
             [0.836025978600521], [0.841470984807897], [0.846831844618015], [0.852108021949363],
             [0.857298989188603], [0.862404227243338], [0.867423225594017], [0.872355482344986],
             [0.877200504274682], [0.881957806884948], [0.886626914449487], [0.891207360061435],
             [0.895698685680048], [0.900100442176505], [0.904412189378826], [0.908633496115883],
             [0.912763940260521], [0.916803108771767], [0.920750597736136], [0.92460601240802], [0.928368967249167],
             [0.932039085967226], [0.935616001553386], [0.939099356319068], [0.942488801931697],
             [0.945783999449539], [0.948984619355586], [0.952090341590516], [0.955100855584692],
             [0.958015860289225], [0.960835064206073], [0.963558185417193], [0.966184951612734],
             [0.968715100118265], [0.971148377921045], [0.973484541695319], [0.975723357826659],
             [0.977864602435316], [0.979908061398614], [0.98185353037236], [0.983700814811277], [0.98544972998846],
             [0.98710010101385], [0.98865176285172], [0.990104560337178], [0.991458348191686], [0.992712991037588],
             [0.993868363411645], [0.994924349777581], [0.99588084453764], [0.996737752043143], [0.997494986604054],
             [0.998152472497548], [0.998710143975583], [0.999167945271476], [0.999525830605479],
             [0.999783764189357], [0.999941720229966], [0.999999682931835]]
    data = NNData(sin_X, sin_Y, .1)
    network.train(data, 10001, order=NNData.Order.RANDOM)
    network.test(data)


def run_XOR():
    network = FFBPNetwork(2, 1)
    network.add_hidden_layer(3)
    data = load_XOR()
    network.train(data, 20000, 1, order=NNData.Order.RANDOM)
    data = load_XOR(0)
    network.test(data)


def run_bias_XOR():
    network = FFBPNetwork(3, 1)
    network.add_hidden_layer(3)
    data = load_bias_XOR()
    network.train(data, 20000, 1, order=NNData.Order.RANDOM)
    data = load_bias_XOR(0)
    network.test(data)


def run_XOR_json():
    xor_data = load_XOR(0.5)
    xor_data.prime_data()
    xor_enc = json.dumps(xor_data, cls=MultiTypeEncoder)
    xor_dec = json.loads(xor_enc, object_hook=multi_type_decoder)
    print(xor_enc)
    print(xor_dec.__dict__)
    network = FFBPNetwork(2, 1)
    network.add_hidden_layer(3)
    network.train(xor_dec, 20001, order=NNData.Order.RANDOM)


def run_sin_json():
    with open("sin_data.txt", "r") as f:
        sin_data = json.load(f, object_hook=multi_type_decoder)
    print(sin_data.__dict__)
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    network.train(sin_data, 10001, order=NNData.Order.RANDOM)


def main():
    run_XOR_json()
    run_sin_json()


if __name__ == "__main__":
    main()

# Sample Run for run_XOR_json Results Below
"""
{"__NNData__": {"_labels": {"__NDarray__": [[0.0], [1.0], [1.0], [0.0]]}, "_features": {"__NDarray__": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]}, "_train_factor": 0.5, "_train_indices": [3, 0], "_test_indices": [1, 2], "_train_pool": {"__deque__": [3, 0]}, "_test_pool": {"__deque__": [1, 2]}}}
{'_labels': array([[0.],
       [1.],
       [1.],
       [0.]]), '_features': array([[0., 0.],
       [1., 0.],
       [0., 1.],
       [1., 1.]]), '_train_factor': 0.5, '_train_indices': [3, 0], '_test_indices': [1, 2], '_train_pool': deque([3, 0]), '_test_pool': deque([1, 2])}
Sample [1. 1.] expected [0.] produced [0.8142208057207088]
Sample [0. 0.] expected [0.] produced [0.7425935426669392]
The epoch: 0  The RMSE: 0.7792306109487471
The epoch: 100  The RMSE: 0.45810912815266797
The epoch: 200  The RMSE: 0.27402526136890243
The epoch: 300  The RMSE: 0.20068610245923124
The epoch: 400  The RMSE: 0.16317190673880863
The epoch: 500  The RMSE: 0.14020489883262724
The epoch: 600  The RMSE: 0.12450691211216589
The epoch: 700  The RMSE: 0.11297902415392772
The epoch: 800  The RMSE: 0.10408072427272977
The epoch: 900  The RMSE: 0.09695731936348037
Sample [1. 1.] expected [0.] produced [0.05345254653158556]
Sample [0. 0.] expected [0.] produced [0.11721520680567554]
The epoch: 1000  The RMSE: 0.0910949489192694
The epoch: 1100  The RMSE: 0.08616462504996834
The epoch: 1200  The RMSE: 0.08194529904868983
The epoch: 1300  The RMSE: 0.07828246981546397
The epoch: 1400  The RMSE: 0.0750645852555893
The epoch: 1500  The RMSE: 0.07220892412766064
The epoch: 1600  The RMSE: 0.06965267940005734
The epoch: 1700  The RMSE: 0.06734725897772068
The epoch: 1800  The RMSE: 0.06525436726210869
The epoch: 1900  The RMSE: 0.0633433944667237
Sample [1. 1.] expected [0.] produced [0.029718535120194102]
Sample [0. 0.] expected [0.] produced [0.08187400467598832]
The epoch: 2000  The RMSE: 0.06158954445104286
The epoch: 2100  The RMSE: 0.05997250482270732
The epoch: 2200  The RMSE: 0.058475431063593576
The epoch: 2300  The RMSE: 0.057084246073599854
The epoch: 2400  The RMSE: 0.055787072576746144
The epoch: 2500  The RMSE: 0.05457381480015613
The epoch: 2600  The RMSE: 0.053435824799189145
The epoch: 2700  The RMSE: 0.05236565733583362
The epoch: 2800  The RMSE: 0.051356852552166564
The epoch: 2900  The RMSE: 0.05040377442646978
Sample [0. 0.] expected [0.] produced [0.0666886860596262]
Sample [1. 1.] expected [0.] produced [0.02129352958131874]
The epoch: 3000  The RMSE: 0.049501491141125634
The epoch: 3100  The RMSE: 0.048645655696632546
The epoch: 3200  The RMSE: 0.04783242573105293
The epoch: 3300  The RMSE: 0.04705838842769934
The epoch: 3400  The RMSE: 0.04632050175026205
The epoch: 3500  The RMSE: 0.045616043093379056
The epoch: 3600  The RMSE: 0.04494256887668069
The epoch: 3700  The RMSE: 0.04429787695657749
The epoch: 3800  The RMSE: 0.04367997703768375
The epoch: 3900  The RMSE: 0.04308706961386614
Sample [1. 1.] expected [0.] produced [0.016867657827631003]
Sample [0. 0.] expected [0.] produced [0.05771447482647772]
The epoch: 4000  The RMSE: 0.042517516890607406
The epoch: 4100  The RMSE: 0.041969827109180416
The epoch: 4200  The RMSE: 0.041442639170393335
The epoch: 4300  The RMSE: 0.040934708212100185
The epoch: 4400  The RMSE: 0.04044489206553193
The epoch: 4500  The RMSE: 0.03997214058991542
The epoch: 4600  The RMSE: 0.039515487683087366
The epoch: 4700  The RMSE: 0.03907404157417337
The epoch: 4800  The RMSE: 0.03864697783065393
The epoch: 4900  The RMSE: 0.038233533602331775
Sample [0. 0.] expected [0.] produced [0.05161410490403551]
Sample [1. 1.] expected [0.] produced [0.014094544219180726]
The epoch: 5000  The RMSE: 0.03783300147880963
The epoch: 5100  The RMSE: 0.03744472409738411
The epoch: 5200  The RMSE: 0.037068090532515245
The epoch: 5300  The RMSE: 0.03670253124514718
The epoch: 5400  The RMSE: 0.036347514332250544
The epoch: 5500  The RMSE: 0.03600254370526695
The epoch: 5600  The RMSE: 0.03566715500074177
The epoch: 5700  The RMSE: 0.035340913981312214
The epoch: 5800  The RMSE: 0.03502341266050099
The epoch: 5900  The RMSE: 0.034714268052432516
Sample [1. 1.] expected [0.] produced [0.012183403562540364]
Sample [0. 0.] expected [0.] produced [0.047117835515061234]
The epoch: 6000  The RMSE: 0.03441312065181017
The epoch: 6100  The RMSE: 0.03411963205226681
The epoch: 6200  The RMSE: 0.03383348360513569
The epoch: 6300  The RMSE: 0.03355437507691928
The epoch: 6400  The RMSE: 0.033282022610668184
The epoch: 6500  The RMSE: 0.03301615844976651
The epoch: 6600  The RMSE: 0.03275652947349749
The epoch: 6700  The RMSE: 0.03250289610339656
The epoch: 6800  The RMSE: 0.032255031200828156
The epoch: 6900  The RMSE: 0.032012720252900716
Sample [0. 0.] expected [0.] produced [0.04362699012324257]
Sample [1. 1.] expected [0.] produced [0.010774202057067592]
The epoch: 7000  The RMSE: 0.031775758820050425
The epoch: 7100  The RMSE: 0.031543953412177285
The epoch: 7200  The RMSE: 0.031317119879770886
The epoch: 7300  The RMSE: 0.031095083304599792
The epoch: 7400  The RMSE: 0.030877677407478
The epoch: 7500  The RMSE: 0.030664743329580753
The epoch: 7600  The RMSE: 0.030456130376554032
The epoch: 7700  The RMSE: 0.030251694501223617
The epoch: 7800  The RMSE: 0.03005129833048909
The epoch: 7900  The RMSE: 0.029854810607382768
Sample [1. 1.] expected [0.] produced [0.009690960360835546]
Sample [0. 0.] expected [0.] produced [0.040813801613020985]
The epoch: 8000  The RMSE: 0.02966210642235578
The epoch: 8100  The RMSE: 0.029473065840173084
The epoch: 8200  The RMSE: 0.02928757447175885
The epoch: 8300  The RMSE: 0.029105522792444105
The epoch: 8400  The RMSE: 0.028926805949792692
The epoch: 8500  The RMSE: 0.028751323583917323
The epoch: 8600  The RMSE: 0.028578979364326147
The epoch: 8700  The RMSE: 0.02840968106533713
The epoch: 8800  The RMSE: 0.028243340066465797
The epoch: 8900  The RMSE: 0.028079871399328243
Sample [0. 0.] expected [0.] produced [0.038484397773380315]
Sample [1. 1.] expected [0.] produced [0.008826883190702564]
The epoch: 9000  The RMSE: 0.027919193566810514
The epoch: 9100  The RMSE: 0.0277612280331516
The epoch: 9200  The RMSE: 0.027605899571211703
The epoch: 9300  The RMSE: 0.027453135588103774
The epoch: 9400  The RMSE: 0.02730286647159324
The epoch: 9500  The RMSE: 0.02715502514243789
The epoch: 9600  The RMSE: 0.027009546996445095
The epoch: 9700  The RMSE: 0.026866369866754003
The epoch: 9800  The RMSE: 0.026725433839856132
The epoch: 9900  The RMSE: 0.02658668107031559
Sample [0. 0.] expected [0.] produced [0.036513725786428614]
Sample [1. 1.] expected [0.] produced [0.008121498343936497]
The epoch: 10000  The RMSE: 0.026450055823731878
The epoch: 10100  The RMSE: 0.026315504553181292
The epoch: 10200  The RMSE: 0.026182975195680264
The epoch: 10300  The RMSE: 0.02605241774932108
The epoch: 10400  The RMSE: 0.02592378384989682
The epoch: 10500  The RMSE: 0.025797026869485262
The epoch: 10600  The RMSE: 0.02567210158596793
The epoch: 10700  The RMSE: 0.02554896442273452
The epoch: 10800  The RMSE: 0.02542757316563438
The epoch: 10900  The RMSE: 0.025307886955985074
Sample [1. 1.] expected [0.] produced [0.007533594612338202]
Sample [0. 0.] expected [0.] produced [0.03481815170570284]
The epoch: 11000  The RMSE: 0.0251898663750365
The epoch: 11100  The RMSE: 0.025073473144339213
The epoch: 11200  The RMSE: 0.02495867033275255
The epoch: 11300  The RMSE: 0.02484542202968903
The epoch: 11400  The RMSE: 0.024733693601656604
The epoch: 11500  The RMSE: 0.02462345131335607
The epoch: 11600  The RMSE: 0.02451466261661723
The epoch: 11700  The RMSE: 0.024407295859636597
The epoch: 11800  The RMSE: 0.024301320360056656
The epoch: 11900  The RMSE: 0.024196706400764857
Sample [1. 1.] expected [0.] produced [0.007034432145131107]
Sample [0. 0.] expected [0.] produced [0.03333921169364368]
The epoch: 12000  The RMSE: 0.024093425160798963
The epoch: 12100  The RMSE: 0.023991448564079077
The epoch: 12200  The RMSE: 0.02389074941610849
The epoch: 12300  The RMSE: 0.02379130131039197
The epoch: 12400  The RMSE: 0.023693078609421384
The epoch: 12500  The RMSE: 0.02359605638844027
The epoch: 12600  The RMSE: 0.0235002103959145
The epoch: 12700  The RMSE: 0.02340551712716705
The epoch: 12800  The RMSE: 0.02331195365261755
The epoch: 12900  The RMSE: 0.023219497739221856
Sample [0. 0.] expected [0.] produced [0.03203432815226168]
Sample [1. 1.] expected [0.] produced [0.006604726071908077]
The epoch: 13000  The RMSE: 0.02312812775228172
The epoch: 13100  The RMSE: 0.023037822592375044
The epoch: 13200  The RMSE: 0.022948561798385954
The epoch: 13300  The RMSE: 0.022860325424751065
The epoch: 13400  The RMSE: 0.02277309404256561
The epoch: 13500  The RMSE: 0.02268684875575336
The epoch: 13600  The RMSE: 0.02260157112828097
The epoch: 13700  The RMSE: 0.022517243243497565
The epoch: 13800  The RMSE: 0.02243384760596525
The epoch: 13900  The RMSE: 0.022351367216265364
Sample [1. 1.] expected [0.] produced [0.006231422520631319]
Sample [0. 0.] expected [0.] produced [0.03087160611883067]
The epoch: 14000  The RMSE: 0.022269785483776606
The epoch: 14100  The RMSE: 0.022189086199707015
The epoch: 14200  The RMSE: 0.022109253601058698
The epoch: 14300  The RMSE: 0.022030272306918154
The epoch: 14400  The RMSE: 0.02195212732273138
The epoch: 14500  The RMSE: 0.0218748040325688
The epoch: 14600  The RMSE: 0.021798288127090636
The epoch: 14700  The RMSE: 0.021722565673478706
The epoch: 14800  The RMSE: 0.021647623081523244
The epoch: 14900  The RMSE: 0.021573447077478203
Sample [0. 0.] expected [0.] produced [0.029827182214800133]
Sample [1. 1.] expected [0.] produced [0.00590265401460136]
The epoch: 15000  The RMSE: 0.021500024689415812
The epoch: 15100  The RMSE: 0.021427343269433123
The epoch: 15200  The RMSE: 0.021355390466696336
The epoch: 15300  The RMSE: 0.02128415417430922
The epoch: 15400  The RMSE: 0.021213622606045295
The epoch: 15500  The RMSE: 0.021143784249556402
The epoch: 15600  The RMSE: 0.0210746278445611
The epoch: 15700  The RMSE: 0.02100614235404826
The epoch: 15800  The RMSE: 0.02093831702906111
The epoch: 15900  The RMSE: 0.020871141354994658
Sample [1. 1.] expected [0.] produced [0.0056116349476548946]
Sample [0. 0.] expected [0.] produced [0.028882048617846694]
The epoch: 16000  The RMSE: 0.02080460500886044
The epoch: 16100  The RMSE: 0.020738697956191474
The epoch: 16200  The RMSE: 0.02067341035187864
The epoch: 16300  The RMSE: 0.020608732564310176
The epoch: 16400  The RMSE: 0.020544655177181036
The epoch: 16500  The RMSE: 0.02048116895985971
The epoch: 16600  The RMSE: 0.020418264891464442
The epoch: 16700  The RMSE: 0.020355934149276413
The epoch: 16800  The RMSE: 0.0202941680799542
The epoch: 16900  The RMSE: 0.020232958227693294
Sample [0. 0.] expected [0.] produced [0.02802157361586509]
Sample [1. 1.] expected [0.] produced [0.005351120276558361]
The epoch: 17000  The RMSE: 0.020172296301159427
The epoch: 17100  The RMSE: 0.020112174193544816
The epoch: 17200  The RMSE: 0.020052583947702665
The epoch: 17300  The RMSE: 0.019993517783862186
The epoch: 17400  The RMSE: 0.01993496806868714
The epoch: 17500  The RMSE: 0.019876927347163325
The epoch: 17600  The RMSE: 0.019819388278739913
The epoch: 17700  The RMSE: 0.01976234369959337
The epoch: 17800  The RMSE: 0.01970578656678599
The epoch: 17900  The RMSE: 0.019649709990733302
Sample [1. 1.] expected [0.] produced [0.005117199860128963]
Sample [0. 0.] expected [0.] produced [0.027233661902078175]
The epoch: 18000  The RMSE: 0.01959410721371617
The epoch: 18100  The RMSE: 0.01953897161655885
The epoch: 18200  The RMSE: 0.019484296697618766
The epoch: 18300  The RMSE: 0.01943007609042739
The epoch: 18400  The RMSE: 0.019376303541574262
The epoch: 18500  The RMSE: 0.019322972932362875
The epoch: 18600  The RMSE: 0.019270078232877793
The epoch: 18700  The RMSE: 0.019217613558937145
The epoch: 18800  The RMSE: 0.019165573123635257
The epoch: 18900  The RMSE: 0.019113951243626894
Sample [1. 1.] expected [0.] produced [0.0049053178348784105]
Sample [0. 0.] expected [0.] produced [0.02650875606323862]
The epoch: 19000  The RMSE: 0.01906274234051159
The epoch: 19100  The RMSE: 0.019011940950949623
The epoch: 19200  The RMSE: 0.018961541703193658
The epoch: 19300  The RMSE: 0.018911539332316867
The epoch: 19400  The RMSE: 0.018861928662504947
The epoch: 19500  The RMSE: 0.018812704608606894
The epoch: 19600  The RMSE: 0.01876386218395447
The epoch: 19700  The RMSE: 0.018715396490218507
The epoch: 19800  The RMSE: 0.018667302717299935
The epoch: 19900  The RMSE: 0.018619576130182256
Sample [1. 1.] expected [0.] produced [0.004712560421539456]
Sample [0. 0.] expected [0.] produced [0.025838844766159517]
The epoch: 20000  The RMSE: 0.018572212100021218
Final Epoch RMSE: 0.018572212100021218
"""

# Sample Run for run_sin_json Results Below
"""
{'_labels': array([[0.        ],
       [0.00999983],
       [0.01999867],
       [0.0299955 ],
       [0.03998933],
       [0.04997917],
       [0.05996401],
       [0.06994285],
       [0.07991469],
       [0.08987855],
       [0.09983342],
       [0.1097783 ],
       [0.11971221],
       [0.12963414],
       [0.13954311],
       [0.14943813],
       [0.15931821],
       [0.16918235],
       [0.17902957],
       [0.18885889],
       [0.19866933],
       [0.2084599 ],
       [0.21822962],
       [0.22797752],
       [0.23770263],
       [0.24740396],
       [0.25708055],
       [0.26673144],
       [0.27635565],
       [0.28595223],
       [0.29552021],
       [0.30505864],
       [0.31456656],
       [0.32404303],
       [0.33348709],
       [0.34289781],
       [0.35227423],
       [0.36161543],
       [0.37092047],
       [0.38018842],
       [0.38941834],
       [0.39860933],
       [0.40776045],
       [0.4168708 ],
       [0.42593947],
       [0.43496553],
       [0.44394811],
       [0.45288629],
       [0.46177918],
       [0.47062589],
       [0.47942554],
       [0.48817725],
       [0.49688014],
       [0.50553334],
       [0.51413599],
       [0.52268723],
       [0.5311862 ],
       [0.53963205],
       [0.54802394],
       [0.55636102],
       [0.56464247],
       [0.57286746],
       [0.58103516],
       [0.58914476],
       [0.59719544],
       [0.60518641],
       [0.61311685],
       [0.62098599],
       [0.62879302],
       [0.63653718],
       [0.64421769],
       [0.65183377],
       [0.65938467],
       [0.66686964],
       [0.67428791],
       [0.68163876],
       [0.68892145],
       [0.69613524],
       [0.70327942],
       [0.71035327],
       [0.71735609],
       [0.72428717],
       [0.73114583],
       [0.73793137],
       [0.74464312],
       [0.75128041],
       [0.75784256],
       [0.76432894],
       [0.77073888],
       [0.77707175],
       [0.78332691],
       [0.78950374],
       [0.79560162],
       [0.80161994],
       [0.8075581 ],
       [0.8134155 ],
       [0.81919157],
       [0.82488571],
       [0.83049737],
       [0.83602598],
       [0.84147098],
       [0.84683184],
       [0.85210802],
       [0.85729899],
       [0.86240423],
       [0.86742323],
       [0.87235548],
       [0.8772005 ],
       [0.88195781],
       [0.88662691],
       [0.89120736],
       [0.89569869],
       [0.90010044],
       [0.90441219],
       [0.9086335 ],
       [0.91276394],
       [0.91680311],
       [0.9207506 ],
       [0.92460601],
       [0.92836897],
       [0.93203909],
       [0.935616  ],
       [0.93909936],
       [0.9424888 ],
       [0.945784  ],
       [0.94898462],
       [0.95209034],
       [0.95510086],
       [0.95801586],
       [0.96083506],
       [0.96355819],
       [0.96618495],
       [0.9687151 ],
       [0.97114838],
       [0.97348454],
       [0.97572336],
       [0.9778646 ],
       [0.97990806],
       [0.98185353],
       [0.98370081],
       [0.98544973],
       [0.9871001 ],
       [0.98865176],
       [0.99010456],
       [0.99145835],
       [0.99271299],
       [0.99386836],
       [0.99492435],
       [0.99588084],
       [0.99673775],
       [0.99749499],
       [0.99815247],
       [0.99871014],
       [0.99916795],
       [0.99952583],
       [0.99978376],
       [0.99994172],
       [0.99999968]]), '_features': array([[0.  ],
       [0.01],
       [0.02],
       [0.03],
       [0.04],
       [0.05],
       [0.06],
       [0.07],
       [0.08],
       [0.09],
       [0.1 ],
       [0.11],
       [0.12],
       [0.13],
       [0.14],
       [0.15],
       [0.16],
       [0.17],
       [0.18],
       [0.19],
       [0.2 ],
       [0.21],
       [0.22],
       [0.23],
       [0.24],
       [0.25],
       [0.26],
       [0.27],
       [0.28],
       [0.29],
       [0.3 ],
       [0.31],
       [0.32],
       [0.33],
       [0.34],
       [0.35],
       [0.36],
       [0.37],
       [0.38],
       [0.39],
       [0.4 ],
       [0.41],
       [0.42],
       [0.43],
       [0.44],
       [0.45],
       [0.46],
       [0.47],
       [0.48],
       [0.49],
       [0.5 ],
       [0.51],
       [0.52],
       [0.53],
       [0.54],
       [0.55],
       [0.56],
       [0.57],
       [0.58],
       [0.59],
       [0.6 ],
       [0.61],
       [0.62],
       [0.63],
       [0.64],
       [0.65],
       [0.66],
       [0.67],
       [0.68],
       [0.69],
       [0.7 ],
       [0.71],
       [0.72],
       [0.73],
       [0.74],
       [0.75],
       [0.76],
       [0.77],
       [0.78],
       [0.79],
       [0.8 ],
       [0.81],
       [0.82],
       [0.83],
       [0.84],
       [0.85],
       [0.86],
       [0.87],
       [0.88],
       [0.89],
       [0.9 ],
       [0.91],
       [0.92],
       [0.93],
       [0.94],
       [0.95],
       [0.96],
       [0.97],
       [0.98],
       [0.99],
       [1.  ],
       [1.01],
       [1.02],
       [1.03],
       [1.04],
       [1.05],
       [1.06],
       [1.07],
       [1.08],
       [1.09],
       [1.1 ],
       [1.11],
       [1.12],
       [1.13],
       [1.14],
       [1.15],
       [1.16],
       [1.17],
       [1.18],
       [1.19],
       [1.2 ],
       [1.21],
       [1.22],
       [1.23],
       [1.24],
       [1.25],
       [1.26],
       [1.27],
       [1.28],
       [1.29],
       [1.3 ],
       [1.31],
       [1.32],
       [1.33],
       [1.34],
       [1.35],
       [1.36],
       [1.37],
       [1.38],
       [1.39],
       [1.4 ],
       [1.41],
       [1.42],
       [1.43],
       [1.44],
       [1.45],
       [1.46],
       [1.47],
       [1.48],
       [1.49],
       [1.5 ],
       [1.51],
       [1.52],
       [1.53],
       [1.54],
       [1.55],
       [1.56],
       [1.57]]), '_train_factor': 0.2, '_train_indices': [1, 8, 15, 17, 21, 24, 34, 39, 41, 44, 47, 48, 49, 53, 54, 56, 61, 66, 69, 80, 82, 83, 87, 90, 97, 110, 120, 136, 145, 146, 148], '_test_indices': [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 40, 42, 43, 45, 46, 50, 51, 52, 55, 57, 58, 59, 60, 62, 63, 64, 65, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 84, 85, 86, 88, 89, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 137, 138, 139, 140, 141, 142, 143, 144, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157], '_train_pool': deque([83, 146, 120, 34, 136, 49, 66, 24, 80, 148, 48, 47, 39, 53, 82, 69, 44, 61, 1, 15, 41, 17, 97, 90, 145, 110, 8, 87, 56, 54, 21]), '_test_pool': deque([7, 155, 127, 143, 141, 121, 62, 151, 156, 93, 63, 105, 144, 113, 100, 103, 154, 84, 94, 18, 123, 134, 92, 68, 67, 35, 116, 19, 98, 75, 76, 22, 43, 132, 33, 71, 65, 95, 89, 51, 124, 91, 23, 50, 101, 16, 46, 111, 58, 102, 138, 106, 78, 4, 13, 14, 119, 99, 125, 149, 152, 153, 27, 0, 85, 131, 5, 117, 157, 57, 142, 10, 88, 72, 42, 3, 37, 147, 40, 29, 81, 36, 30, 133, 79, 55, 77, 130, 20, 150, 25, 45, 12, 11, 129, 9, 2, 114, 59, 115, 96, 38, 139, 86, 32, 118, 107, 52, 122, 126, 108, 112, 70, 6, 60, 135, 64, 73, 31, 28, 128, 109, 137, 104, 26, 140, 74])}
Sample [0.69] expected [0.63653718] produced [0.7606026170081999]
Sample [1.1] expected [0.89120736] produced [0.7836968356008933]
Sample [0.01] expected [0.00999983] produced [0.7129587394525504]
Sample [0.34] expected [0.33348709] produced [0.7360516687255625]
Sample [0.54] expected [0.51413599] produced [0.7489341242532744]
Sample [0.53] expected [0.50553334] produced [0.7478083289500035]
Sample [0.41] expected [0.39860933] produced [0.7392513575372303]
Sample [0.56] expected [0.5311862] produced [0.7486277747890361]
Sample [0.87] expected [0.76432894] produced [0.7671673179616665]
Sample [1.2] expected [0.93203909] produced [0.784607778066126]
Sample [1.36] expected [0.9778646] produced [0.7923781884203465]
Sample [0.17] expected [0.16918235] produced [0.7218489962729306]
Sample [1.48] expected [0.99588084] produced [0.7968480895281885]
Sample [0.48] expected [0.46177918] produced [0.7430353937284303]
Sample [0.9] expected [0.78332691] produced [0.7684311923169832]
Sample [0.82] expected [0.73114583] produced [0.7638652844930027]
Sample [0.8] expected [0.71735609] produced [0.7626181183686651]
Sample [0.61] expected [0.57286746] produced [0.7508454205766756]
Sample [1.46] expected [0.99386836] produced [0.7953480678835656]
Sample [0.15] expected [0.14943813] produced [0.7192613307535537]
Sample [0.97] expected [0.82488571] produced [0.7711457719817965]
Sample [0.47] expected [0.45288629] produced [0.7408240639719486]
Sample [0.39] expected [0.38018842] produced [0.734841936622905]
Sample [0.08] expected [0.07991469] produced [0.7120871424298528]
Sample [0.49] expected [0.47062589] produced [0.7397402991723848]
Sample [0.44] expected [0.42593947] produced [0.7358737625807836]
Sample [0.66] expected [0.61311685] produced [0.7494482540165852]
Sample [0.21] expected [0.2084599] produced [0.7190869438879087]
Sample [0.83] expected [0.73793137] produced [0.7583030778314142]
Sample [1.45] expected [0.99271299] produced [0.7890221380016427]
Sample [0.24] expected [0.23770263] produced [0.720677807059838]
The epoch: 0  The RMSE: 0.31997658757964764
The epoch: 100  The RMSE: 0.2712392479081072
The epoch: 200  The RMSE: 0.2642731224919536
The epoch: 300  The RMSE: 0.24068041964367143
The epoch: 400  The RMSE: 0.19609663952595247
The epoch: 500  The RMSE: 0.1580338257691233
The epoch: 600  The RMSE: 0.13147326000977394
The epoch: 700  The RMSE: 0.11260647173818618
The epoch: 800  The RMSE: 0.09862617194754057
The epoch: 900  The RMSE: 0.08786935521272475
Sample [0.9] expected [0.78332691] produced [0.7362430658020537]
Sample [0.39] expected [0.38018842] produced [0.42791913689711625]
Sample [0.54] expected [0.51413599] produced [0.5431388135055106]
Sample [0.48] expected [0.46177918] produced [0.4985255695576984]
Sample [0.97] expected [0.82488571] produced [0.7594055321929138]
Sample [0.69] expected [0.63653718] produced [0.6396172704678156]
Sample [0.01] expected [0.00999983] produced [0.15854447724052065]
Sample [0.49] expected [0.47062589] produced [0.5059819563321017]
Sample [0.53] expected [0.50553334] produced [0.5354961142166241]
Sample [1.36] expected [0.9778646] produced [0.8375040579484538]
Sample [0.83] expected [0.73793137] produced [0.7086030652375276]
Sample [0.66] expected [0.61311685] produced [0.6221586167362246]
Sample [1.2] expected [0.93203909] produced [0.8141003319562552]
Sample [0.8] expected [0.71735609] produced [0.6958554746107037]
Sample [0.41] expected [0.39860933] produced [0.4440888936876695]
Sample [0.24] expected [0.23770263] produced [0.3081351738854666]
Sample [0.82] expected [0.73114583] produced [0.7044340355154555]
Sample [1.48] expected [0.99588084] produced [0.850678576860542]
Sample [0.08] expected [0.07991469] produced [0.19750958523450626]
Sample [0.17] expected [0.16918235] produced [0.25637489363328697]
Sample [0.87] expected [0.76432894] produced [0.7250045430002379]
Sample [1.46] expected [0.99386836] produced [0.8488462597269336]
Sample [0.47] expected [0.45288629] produced [0.4914311351293989]
Sample [1.45] expected [0.99271299] produced [0.8479990624763684]
Sample [0.44] expected [0.42593947] produced [0.4682258726962766]
Sample [1.1] expected [0.89120736] produced [0.79451628660983]
Sample [0.15] expected [0.14943813] produced [0.24277660117975164]
Sample [0.21] expected [0.2084599] produced [0.2856548558965103]
Sample [0.34] expected [0.33348709] produced [0.38760359098077135]
Sample [0.61] expected [0.57286746] produced [0.5912278178451715]
Sample [0.56] expected [0.5311862] produced [0.5573991779974089]
The epoch: 1000  The RMSE: 0.07935957460733242
The epoch: 1100  The RMSE: 0.0724823418591285
The epoch: 1200  The RMSE: 0.06683994291998181
The epoch: 1300  The RMSE: 0.06214811529949651
The epoch: 1400  The RMSE: 0.058206476049511825
The epoch: 1500  The RMSE: 0.054876536264076126
The epoch: 1600  The RMSE: 0.05203493940136451
The epoch: 1700  The RMSE: 0.04959825215077192
The epoch: 1800  The RMSE: 0.04749694294206733
The epoch: 1900  The RMSE: 0.04567736139077075
Sample [0.49] expected [0.47062589] produced [0.484457779888223]
Sample [0.41] expected [0.39860933] produced [0.4057437133539185]
Sample [0.44] expected [0.42593947] produced [0.43543425023543897]
Sample [0.47] expected [0.45288629] produced [0.46488959960606924]
Sample [0.66] expected [0.61311685] produced [0.6328535657162517]
Sample [0.82] expected [0.73114583] produced [0.7356213687232451]
Sample [0.61] expected [0.57286746] produced [0.5928267544136078]
Sample [0.17] expected [0.16918235] produced [0.1911595371002993]
Sample [0.69] expected [0.63653718] produced [0.6547539516993165]
Sample [0.87] expected [0.76432894] produced [0.7601845499861617]
Sample [0.01] expected [0.00999983] produced [0.09966319376698712]
Sample [0.54] expected [0.51413599] produced [0.5309908068241375]
Sample [1.45] expected [0.99271299] produced [0.8956803620396568]
Sample [0.56] expected [0.5311862] produced [0.5492896781058219]
Sample [1.2] expected [0.93203909] produced [0.861295533518683]
Sample [0.34] expected [0.33348709] produced [0.33676454214215146]
Sample [0.9] expected [0.78332691] produced [0.7736884790780663]
Sample [0.53] expected [0.50553334] produced [0.5219825790461149]
Sample [1.46] expected [0.99386836] produced [0.8967556031792538]
Sample [0.83] expected [0.73793137] produced [0.7408637406206453]
Sample [1.1] expected [0.89120736] produced [0.8398327779573409]
Sample [0.48] expected [0.46177918] produced [0.47462895396794225]
Sample [0.97] expected [0.82488571] produced [0.8011989153045882]
Sample [0.15] expected [0.14943813] produced [0.17723971282810846]
Sample [0.21] expected [0.2084599] produced [0.22153369571089027]
Sample [1.36] expected [0.9778646] produced [0.8860265309231552]
Sample [0.08] expected [0.07991469] produced [0.13404085217116168]
Sample [0.24] expected [0.23770263] produced [0.2461101872028828]
Sample [0.8] expected [0.71735609] produced [0.7249857373954296]
Sample [1.48] expected [0.99588084] produced [0.8987686329150496]
Sample [0.39] expected [0.38018842] produced [0.3860307195207271]
The epoch: 2000  The RMSE: 0.044092464474050116
The epoch: 2100  The RMSE: 0.04270882806862628
The epoch: 2200  The RMSE: 0.041493229192503436
The epoch: 2300  The RMSE: 0.040423390000150584
The epoch: 2400  The RMSE: 0.039476247844907664
The epoch: 2500  The RMSE: 0.038635278881690396
The epoch: 2600  The RMSE: 0.037885336966438435
The epoch: 2700  The RMSE: 0.037214105919468284
The epoch: 2800  The RMSE: 0.03661232197333782
The epoch: 2900  The RMSE: 0.036068889938792176
Sample [0.53] expected [0.50553334] produced [0.516261021504137]
Sample [0.8] expected [0.71735609] produced [0.7344949136093069]
Sample [0.08] expected [0.07991469] produced [0.11989015634141593]
Sample [1.2] expected [0.93203909] produced [0.8787967341279733]
Sample [0.01] expected [0.00999983] produced [0.08771715938364945]
Sample [0.24] expected [0.23770263] produced [0.22911303160926855]
Sample [0.49] expected [0.47062589] produced [0.47566928197954195]
Sample [0.48] expected [0.46177918] produced [0.4653507252620047]
Sample [0.69] expected [0.63653718] produced [0.6591434210066576]
Sample [0.9] expected [0.78332691] produced [0.7867670579258756]
Sample [1.48] expected [0.99588084] produced [0.9165423617933222]
Sample [0.87] expected [0.76432894] produced [0.7726019245106465]
Sample [0.21] expected [0.2084599] produced [0.2047264346910135]
Sample [1.46] expected [0.99386836] produced [0.9147241708040303]
Sample [0.15] expected [0.14943813] produced [0.1613892521484563]
Sample [1.45] expected [0.99271299] produced [0.9138039687754274]
Sample [0.82] expected [0.73114583] produced [0.7462926863942924]
Sample [0.54] expected [0.51413599] produced [0.526214741600712]
Sample [0.56] expected [0.5311862] produced [0.5456506092013821]
Sample [0.66] expected [0.61311685] produced [0.6352356316701796]
Sample [0.34] expected [0.33348709] produced [0.3211974937176509]
Sample [0.61] expected [0.57286746] produced [0.5921101541026796]
Sample [0.41] expected [0.39860933] produced [0.3924360452424649]
Sample [0.47] expected [0.45288629] produced [0.45487564189525165]
Sample [1.1] expected [0.89120736] produced [0.8563737671530994]
Sample [0.97] expected [0.82488571] produced [0.8158403385535931]
Sample [0.17] expected [0.16918235] produced [0.17500879976983633]
Sample [1.36] expected [0.9778646] produced [0.9038164518143599]
Sample [0.44] expected [0.42593947] produced [0.4239529441464901]
Sample [0.39] expected [0.38018842] produced [0.3720278776195733]
Sample [0.83] expected [0.73793137] produced [0.751843481843581]
The epoch: 3000  The RMSE: 0.03557826886735729
The epoch: 3100  The RMSE: 0.035131925952435236
The epoch: 3200  The RMSE: 0.03472637934619057
The epoch: 3300  The RMSE: 0.03435503986514179
The epoch: 3400  The RMSE: 0.03401481101356657
The epoch: 3500  The RMSE: 0.03370157865617487
The epoch: 3600  The RMSE: 0.03341276285604136
The epoch: 3700  The RMSE: 0.03314549040160343
The epoch: 3800  The RMSE: 0.03289746972290137
The epoch: 3900  The RMSE: 0.032666889631209825
Sample [1.48] expected [0.99588084] produced [0.9257859045927158]
Sample [0.15] expected [0.14943813] produced [0.15669342249989696]
Sample [0.61] expected [0.57286746] produced [0.5910282103059427]
Sample [0.83] expected [0.73793137] produced [0.75606841062622]
Sample [0.56] expected [0.5311862] produced [0.5428333870762091]
Sample [0.49] expected [0.47062589] produced [0.47120043673785245]
Sample [0.87] expected [0.76432894] produced [0.7776491015734288]
Sample [0.39] expected [0.38018842] produced [0.36593769604696064]
Sample [0.01] expected [0.00999983] produced [0.08469244569447022]
Sample [1.1] expected [0.89120736] produced [0.8643258209923947]
Sample [0.82] expected [0.73114583] produced [0.7501500642590481]
Sample [0.41] expected [0.39860933] produced [0.38681191511448265]
Sample [0.17] expected [0.16918235] produced [0.16999764861380523]
Sample [0.53] expected [0.50553334] produced [0.5125782824126325]
Sample [0.34] expected [0.33348709] produced [0.31509269651726723]
Sample [0.69] expected [0.63653718] produced [0.6600035859806072]
Sample [0.97] expected [0.82488571] produced [0.8223926583598644]
Sample [0.66] expected [0.61311685] produced [0.6350935404950523]
Sample [0.21] expected [0.2084599] produced [0.1991825768359475]
Sample [0.54] expected [0.51413599] produced [0.5225271189034196]
Sample [0.08] expected [0.07991469] produced [0.11597185117285957]
Sample [1.36] expected [0.9778646] produced [0.9126798042400746]
Sample [0.48] expected [0.46177918] produced [0.460544631300383]
Sample [0.44] expected [0.42593947] produced [0.4183410964543017]
Sample [1.45] expected [0.99271299] produced [0.9228419250844636]
Sample [1.46] expected [0.99386836] produced [0.9238645639136692]
Sample [0.24] expected [0.23770263] produced [0.2234362142674379]
Sample [0.47] expected [0.45288629] produced [0.4503643910162182]
Sample [0.8] expected [0.71735609] produced [0.7382313012003715]
Sample [0.9] expected [0.78332691] produced [0.7925270932849325]
Sample [1.2] expected [0.93203909] produced [0.8873554446473185]
The epoch: 4000  The RMSE: 0.03245145581520629
The epoch: 4100  The RMSE: 0.03225045721116812
The epoch: 4200  The RMSE: 0.032062014906143146
The epoch: 4300  The RMSE: 0.031884520691974734
The epoch: 4400  The RMSE: 0.031717999798965245
The epoch: 4500  The RMSE: 0.0315617521296347
The epoch: 4600  The RMSE: 0.03141332850583501
The epoch: 4700  The RMSE: 0.031273060232523305
The epoch: 4800  The RMSE: 0.03114007794529193
The epoch: 4900  The RMSE: 0.03101306791594767
Sample [0.97] expected [0.82488571] produced [0.8260884183735946]
Sample [1.48] expected [0.99588084] produced [0.9313153236364403]
Sample [0.83] expected [0.73793137] produced [0.7580870202723194]
Sample [0.15] expected [0.14943813] produced [0.15542208566518068]
Sample [0.53] expected [0.50553334] produced [0.5105391342887372]
Sample [0.56] expected [0.5311862] produced [0.5409846381640335]
Sample [1.2] expected [0.93203909] produced [0.892292900142286]
Sample [0.44] expected [0.42593947] produced [0.4159115350227814]
Sample [0.48] expected [0.46177918] produced [0.4583556528569975]
Sample [0.34] expected [0.33348709] produced [0.31270466901980803]
Sample [1.46] expected [0.99386836] produced [0.9294736994609151]
Sample [0.49] expected [0.47062589] produced [0.46913975028230603]
Sample [0.9] expected [0.78332691] produced [0.7956098976427427]
Sample [0.54] expected [0.51413599] produced [0.5210579665702656]
Sample [0.17] expected [0.16918235] produced [0.16875901385678602]
Sample [1.1] expected [0.89120736] produced [0.8690320416819988]
Sample [1.36] expected [0.9778646] produced [0.9183223606931319]
Sample [0.24] expected [0.23770263] produced [0.22170840030810074]
Sample [0.87] expected [0.76432894] produced [0.7806004503912145]
Sample [1.45] expected [0.99271299] produced [0.9285501057039645]
Sample [0.01] expected [0.00999983] produced [0.08439645806877769]
Sample [0.8] expected [0.71735609] produced [0.7401056813762705]
Sample [0.39] expected [0.38018842] produced [0.3636230606285278]
Sample [0.82] expected [0.73114583] produced [0.7524067075388243]
Sample [0.69] expected [0.63653718] produced [0.6603019860212679]
Sample [0.21] expected [0.2084599] produced [0.19771544979927116]
Sample [0.61] expected [0.57286746] produced [0.5897979861967724]
Sample [0.08] expected [0.07991469] produced [0.11533481637893467]
Sample [0.47] expected [0.45288629] produced [0.447716061225382]
Sample [0.66] expected [0.61311685] produced [0.6348308649680356]
Sample [0.41] expected [0.39860933] produced [0.3841528609323322]
The epoch: 5000  The RMSE: 0.030893700509432177
The epoch: 5100  The RMSE: 0.030779421631422995
The epoch: 5200  The RMSE: 0.030670699213069936
The epoch: 5300  The RMSE: 0.0305660585465155
The epoch: 5400  The RMSE: 0.030467237986459658
The epoch: 5500  The RMSE: 0.03037282002545056
The epoch: 5600  The RMSE: 0.030281956740793
The epoch: 5700  The RMSE: 0.0301946935279973
The epoch: 5800  The RMSE: 0.030110936295662635
The epoch: 5900  The RMSE: 0.030030909769408184
Sample [0.48] expected [0.46177918] produced [0.4566371834666942]
Sample [0.49] expected [0.47062589] produced [0.46723177198097116]
Sample [0.83] expected [0.73793137] produced [0.7589866307004963]
Sample [0.21] expected [0.2084599] produced [0.19717908292610317]
Sample [0.69] expected [0.63653718] produced [0.6595675993017573]
Sample [0.34] expected [0.33348709] produced [0.31123931013773976]
Sample [0.41] expected [0.39860933] produced [0.38259669703001953]
Sample [0.44] expected [0.42593947] produced [0.4142708673414244]
Sample [1.45] expected [0.99271299] produced [0.932134805028437]
Sample [0.82] expected [0.73114583] produced [0.7530737491437706]
Sample [0.66] expected [0.61311685] produced [0.6341698449464941]
Sample [0.87] expected [0.76432894] produced [0.7814701085334469]
Sample [0.24] expected [0.23770263] produced [0.22078803073634085]
Sample [1.2] expected [0.93203909] produced [0.8955030468197924]
Sample [0.53] expected [0.50553334] produced [0.5090081907345039]
Sample [1.46] expected [0.99386836] produced [0.9331048518763104]
Sample [0.54] expected [0.51413599] produced [0.5193919733798676]
Sample [1.36] expected [0.9778646] produced [0.921810788146911]
Sample [1.1] expected [0.89120736] produced [0.8718537458702169]
Sample [0.47] expected [0.45288629] produced [0.44629243003306146]
Sample [1.48] expected [0.99588084] produced [0.9351153300739072]
Sample [0.61] expected [0.57286746] produced [0.5889494265427591]
Sample [0.08] expected [0.07991469] produced [0.11558494990173844]
Sample [0.17] expected [0.16918235] produced [0.16856164421097905]
Sample [0.01] expected [0.00999983] produced [0.08472969534864692]
Sample [0.9] expected [0.78332691] produced [0.7970839315374277]
Sample [0.15] expected [0.14943813] produced [0.1553477193961982]
Sample [0.97] expected [0.82488571] produced [0.8282662592358296]
Sample [0.56] expected [0.5311862] produced [0.5397091373863204]
Sample [0.39] expected [0.38018842] produced [0.36187689715963933]
Sample [0.8] expected [0.71735609] produced [0.7404847893382813]
The epoch: 6000  The RMSE: 0.029953675623888617
The epoch: 6100  The RMSE: 0.029879227430154854
The epoch: 6200  The RMSE: 0.029807825380158727
The epoch: 6300  The RMSE: 0.02973905469915225
The epoch: 6400  The RMSE: 0.029672536906549248
The epoch: 6500  The RMSE: 0.02960804978077267
The epoch: 6600  The RMSE: 0.029546321459231223
The epoch: 6700  The RMSE: 0.02948617268376283
The epoch: 6800  The RMSE: 0.02942874373796432
The epoch: 6900  The RMSE: 0.029372686490607487
Sample [0.8] expected [0.71735609] produced [0.7407457159882098]
Sample [0.21] expected [0.2084599] produced [0.19714707489245237]
Sample [0.39] expected [0.38018842] produced [0.3608674244003508]
Sample [1.1] expected [0.89120736] produced [0.8737821194225734]
Sample [0.01] expected [0.00999983] produced [0.08525172138298052]
Sample [0.69] expected [0.63653718] produced [0.6592109241026046]
Sample [0.49] expected [0.47062589] produced [0.46593026425481493]
Sample [0.34] expected [0.33348709] produced [0.3105974109481441]
Sample [1.46] expected [0.99386836] produced [0.935806877004948]
Sample [0.97] expected [0.82488571] produced [0.8297144629399305]
Sample [0.54] expected [0.51413599] produced [0.5182962288630587]
Sample [0.56] expected [0.5311862] produced [0.5386360061986555]
Sample [0.08] expected [0.07991469] produced [0.1159593724899204]
Sample [1.48] expected [0.99588084] produced [0.9377294122263355]
Sample [1.2] expected [0.93203909] produced [0.89790733687112]
Sample [0.9] expected [0.78332691] produced [0.7981132438361825]
Sample [0.17] expected [0.16918235] produced [0.1686610758312983]
Sample [0.83] expected [0.73793137] produced [0.759600344494957]
Sample [0.15] expected [0.14943813] produced [0.15554354711334495]
Sample [0.53] expected [0.50553334] produced [0.5078544922782631]
Sample [0.44] expected [0.42593947] produced [0.4131311250861593]
Sample [0.48] expected [0.46177918] produced [0.45552156658198184]
Sample [1.36] expected [0.9778646] produced [0.9244121577850191]
Sample [0.41] expected [0.39860933] produced [0.3818207148655657]
Sample [0.82] expected [0.73114583] produced [0.7536561918469776]
Sample [0.87] expected [0.76432894] produced [0.7824978046280657]
Sample [0.61] expected [0.57286746] produced [0.5877169581565226]
Sample [0.66] expected [0.61311685] produced [0.6334468552433016]
Sample [0.47] expected [0.45288629] produced [0.44473854395903334]
Sample [1.45] expected [0.99271299] produced [0.9347595064782831]
Sample [0.24] expected [0.23770263] produced [0.22067802648145357]
The epoch: 7000  The RMSE: 0.029318489403490956
The epoch: 7100  The RMSE: 0.02926592371259283
The epoch: 7200  The RMSE: 0.029213880145222172
The epoch: 7300  The RMSE: 0.029165527125192773
The epoch: 7400  The RMSE: 0.029117572965125328
The epoch: 7500  The RMSE: 0.029070941210002792
The epoch: 7600  The RMSE: 0.029025902242565987
The epoch: 7700  The RMSE: 0.028981509124399004
The epoch: 7800  The RMSE: 0.028939032185029124
The epoch: 7900  The RMSE: 0.028896787024156965
Sample [1.36] expected [0.9778646] produced [0.9263949479813549]
Sample [0.21] expected [0.2084599] produced [0.19738424835353682]
Sample [0.48] expected [0.46177918] produced [0.454804030463133]
Sample [0.49] expected [0.47062589] produced [0.46539994042189564]
Sample [0.97] expected [0.82488571] produced [0.830870679784587]
Sample [0.53] expected [0.50553334] produced [0.5072451074960985]
Sample [0.47] expected [0.45288629] produced [0.44427383471282617]
Sample [0.8] expected [0.71735609] produced [0.7411650008287357]
Sample [0.83] expected [0.73793137] produced [0.7599771466947787]
Sample [0.61] expected [0.57286746] produced [0.5869346580467285]
Sample [0.15] expected [0.14943813] produced [0.15586400776831677]
Sample [0.69] expected [0.63653718] produced [0.6586698007735484]
Sample [1.1] expected [0.89120736] produced [0.8751477117624641]
Sample [1.48] expected [0.99588084] produced [0.9397647765949549]
Sample [1.45] expected [0.99271299] produced [0.9368369344902087]
Sample [0.9] expected [0.78332691] produced [0.798702723791052]
Sample [0.82] expected [0.73114583] produced [0.753639587890416]
Sample [0.17] expected [0.16918235] produced [0.16883956482350068]
Sample [0.54] expected [0.51413599] produced [0.5171111712987378]
Sample [0.39] expected [0.38018842] produced [0.3601027965999171]
Sample [0.56] expected [0.5311862] produced [0.5375937202842728]
Sample [0.41] expected [0.39860933] produced [0.38081661439423253]
Sample [0.01] expected [0.00999983] produced [0.0857877648798569]
Sample [1.2] expected [0.93203909] produced [0.8995802024615914]
Sample [1.46] expected [0.99386836] produced [0.9378720279314274]
Sample [0.24] expected [0.23770263] produced [0.22073584653189932]
Sample [0.44] expected [0.42593947] produced [0.41250325614226585]
Sample [0.87] expected [0.76432894] produced [0.7831276621348758]
Sample [0.08] expected [0.07991469] produced [0.1164675431483856]
Sample [0.34] expected [0.33348709] produced [0.31026834990130897]
Sample [0.66] expected [0.61311685] produced [0.633140608568714]
The epoch: 8000  The RMSE: 0.028857280862657615
The epoch: 8100  The RMSE: 0.028817986085328956
The epoch: 8200  The RMSE: 0.02877985031536112
The epoch: 8300  The RMSE: 0.02874257312245664
The epoch: 8400  The RMSE: 0.02870619806532503
The epoch: 8500  The RMSE: 0.028670802331568376
The epoch: 8600  The RMSE: 0.028635794898483894
The epoch: 8700  The RMSE: 0.028602464941131993
The epoch: 8800  The RMSE: 0.028569085554345904
The epoch: 8900  The RMSE: 0.02853730895851778
Sample [0.41] expected [0.39860933] produced [0.38044099699995754]
Sample [0.15] expected [0.14943813] produced [0.15627986259330687]
Sample [0.82] expected [0.73114583] produced [0.754031651316055]
Sample [1.46] expected [0.99386836] produced [0.9395416974146208]
Sample [0.56] expected [0.5311862] produced [0.5371183278060965]
Sample [0.17] expected [0.16918235] produced [0.16922015174331725]
Sample [1.1] expected [0.89120736] produced [0.8765414692183336]
Sample [0.97] expected [0.82488571] produced [0.8315920928910457]
Sample [0.24] expected [0.23770263] produced [0.2208143806157996]
Sample [1.36] expected [0.9778646] produced [0.9280338497802405]
Sample [0.39] expected [0.38018842] produced [0.36000228048774846]
Sample [1.2] expected [0.93203909] produced [0.901216822342217]
Sample [0.08] expected [0.07991469] produced [0.11700302043011172]
Sample [0.66] expected [0.61311685] produced [0.6329146338037992]
Sample [0.54] expected [0.51413599] produced [0.516878735404692]
Sample [0.21] expected [0.2084599] produced [0.19760044284051786]
Sample [0.53] expected [0.50553334] produced [0.5065703701524338]
Sample [1.45] expected [0.99271299] produced [0.9386060699859837]
Sample [0.44] expected [0.42593947] produced [0.4120951548270976]
Sample [0.87] expected [0.76432894] produced [0.7837596216770559]
Sample [0.47] expected [0.45288629] produced [0.44371581347005745]
Sample [1.48] expected [0.99588084] produced [0.9416098599458509]
Sample [0.01] expected [0.00999983] produced [0.08636397105682335]
Sample [0.48] expected [0.46177918] produced [0.4543603534378716]
Sample [0.61] expected [0.57286746] produced [0.5868424317242827]
Sample [0.9] expected [0.78332691] produced [0.7995713035473743]
Sample [0.49] expected [0.47062589] produced [0.4647699272002991]
Sample [0.83] expected [0.73793137] produced [0.7604252676192912]
Sample [0.69] expected [0.63653718] produced [0.658642965481887]
Sample [0.8] expected [0.71735609] produced [0.7410807538450163]
Sample [0.34] expected [0.33348709] produced [0.30985209698719246]
The epoch: 9000  The RMSE: 0.02850548923420741
The epoch: 9100  The RMSE: 0.028475255139921687
The epoch: 9200  The RMSE: 0.02844511438255858
The epoch: 9300  The RMSE: 0.028416252222096774
The epoch: 9400  The RMSE: 0.02838763645213031
The epoch: 9500  The RMSE: 0.028359558309657695
The epoch: 9600  The RMSE: 0.028331726649351036
The epoch: 9700  The RMSE: 0.02830525805400641
The epoch: 9800  The RMSE: 0.028278966318295774
The epoch: 9900  The RMSE: 0.028252462414269368
Sample [0.21] expected [0.2084599] produced [0.19772955364295933]
Sample [0.82] expected [0.73114583] produced [0.7541674599665986]
Sample [0.54] expected [0.51413599] produced [0.5159837152299124]
Sample [0.48] expected [0.46177918] produced [0.45333113724853596]
Sample [0.41] expected [0.39860933] produced [0.3799896537559198]
Sample [0.66] expected [0.61311685] produced [0.6322127832312818]
Sample [0.17] expected [0.16918235] produced [0.169495231425715]
Sample [0.34] expected [0.33348709] produced [0.3096966804661183]
Sample [1.1] expected [0.89120736] produced [0.8775667437910762]
Sample [0.47] expected [0.45288629] produced [0.44297835555271226]
Sample [0.9] expected [0.78332691] produced [0.7998666351316275]
Sample [0.56] expected [0.5311862] produced [0.5365612801509652]
Sample [0.15] expected [0.14943813] produced [0.15660546099234038]
Sample [0.24] expected [0.23770263] produced [0.22092878349088965]
Sample [0.69] expected [0.63653718] produced [0.6582351396350725]
Sample [0.61] expected [0.57286746] produced [0.5857703161198606]
Sample [1.48] expected [0.99588084] produced [0.9428516350385945]
Sample [1.36] expected [0.9778646] produced [0.9293120769095315]
Sample [0.39] expected [0.38018842] produced [0.3595058037314723]
Sample [0.49] expected [0.47062589] produced [0.46410301552898625]
Sample [1.2] expected [0.93203909] produced [0.9023384760117275]
Sample [0.08] expected [0.07991469] produced [0.11742342167308349]
Sample [0.53] expected [0.50553334] produced [0.505990747994157]
Sample [0.01] expected [0.00999983] produced [0.08681044346172505]
Sample [1.45] expected [0.99271299] produced [0.9399716643009186]
Sample [0.97] expected [0.82488571] produced [0.8324248172460559]
Sample [0.8] expected [0.71735609] produced [0.74137597341153]
Sample [0.44] expected [0.42593947] produced [0.4113792611201061]
Sample [1.46] expected [0.99386836] produced [0.9409923278766635]
Sample [0.87] expected [0.76432894] produced [0.7839924640916127]
Sample [0.83] expected [0.73793137] produced [0.7604993094082596]
The epoch: 10000  The RMSE: 0.028228449868957728
Final Epoch RMSE: 0.028228449868957728
"""
