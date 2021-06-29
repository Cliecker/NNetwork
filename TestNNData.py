# First Unit Test-Assignment 1
"""
print("\nTest #1")
test = NNData()
try:
    test.load_data([0, 1, 2], [9, 1])
except DataMismatchError:
    print("DataMismatchError Occurred with NNData.load_data([0, 1, 2], [9, 1])")
    print(f"self._features is set to {test._features}")
    print(f"self._labels is set to {test._labels}")

print("\nTest #2")
test2 = NNData()
try:
    test2.load_data([0, 2, "yes"], ["no", 2, 2])
except ValueError:
    print("ValueError Occurred with NNData.load_data([0, 2, \"yes\"], [\"no\", 2, 2])")
    print(f"self._features is set to {test2._features}")
    print(f"self._labels is set to {test2._labels}")

print("\nTest #3")
try:
    NNData([0, 1, 2], [9, 1], 1)
except DataMismatchError as instance:
    test3 = instance.args[0]
    print("DataMismatchError Occurred with NNData([0, 1, 2], [9, 1], 1)")
    print(f"self._features is set to {test3._features}")
    print(f"self._labels is set to {test3._labels}")

print("\nTest #4")
try:
    NNData([0, "es", 2], [9, "string", "asd"], 1)
except ValueError as instance:
    test4 = instance.args[0]
    print("ValueError Occurred with NNData([0, \"es\", 2], [9, \"string\", \"asd\"], 1)")
    print(f"self._features is set to {test4._features}")
    print(f"self._labels is set to {test4._labels}")

print("\nTest #5")
test5 = NNData([0, 1, 2], [0, 1, 2], -1)
print(f"Set training factor to zero when a negative (ie. -1) is passed: {test5._train_factor}")

print("\nTest #6")
test6 = NNData([0, 1, 2], [0, 1, 2], 5)
print(f"Set training factor to one when a number greater than one (ie. 2) is passed: {test6._train_factor}")
"""

# Second Unit Test-Assignment 2 Checkpoint 1
"""
errors = False
try:
    # Create a valid small and large dataset to be used later
    x = list(range(10))
    y = x
    our_data_0 = NNData(x, y)
    print(our_data_0._features)
    x = list(range(100))
    y = x
    our_big_data = NNData(x, y, .5)
except:
    print("There are errors that likely come from __init__ or a method called by __init__")
    errors = True

# Test split_set to make sure the correct number of samples are in
# each set, and that the indices do not overlap.
try:
    our_data_0.split_set(.3)
    assert len(our_data_0._train_indices) == 3
    assert len(our_data_0._test_indices) == 7
    assert (list(set(our_data_0._train_indices + our_data_0._test_indices))) == list(range(10))
except:
    print("There are errors that likely come from split_set")
    errors = True  # Summary
if errors:
    print("You have one or more errors.  Please fix them before submitting")
else:
    print("No errors were identified by the unit test.")
    print("You should still double check that your code meets spec.")
    print("You should also check that PyCharm does not identify any PEP-8 issues.")
"""

# Third Unit Test-Assignment 2
"""
    errors = False
    try:
        # Create a valid small and large dataset to be used later
        x = [[i] for i in range(10)]
        y = x
        our_data_0 = NNData(x, y)
        x = [[i] for i in range(100)]
        y = x
        our_big_data = NNData(x, y, .5)

        # Try loading lists of different sizes
        y = [[1]]
        try:
            our_bad_data = NNData()
            our_bad_data.load_data(x, y)
            raise Exception
        except DataMismatchError:
            pass
        except:
            raise Exception

        # Create a dataset that can be used to make sure the
        # features and labels are not confused
        x = [[1.0], [2.0], [3.0], [4.0]]
        y = [[.1], [.2], [.3], [.4]]
        our_data_1 = NNData(x, y, .5)

    except:
        print("There are errors that likely come from __init__ or a method called by __init__")
        errors = True

    # Test split_set to make sure the correct number of examples are in
    # each set, and that the indices do not overlap.
    try:
        our_data_0.split_set(.3)
        print(f"Train Indices:{our_data_0._train_indices}")
        print(f"Test Indices:{our_data_0._test_indices}")

        assert len(our_data_0._train_indices) == 3
        assert len(our_data_0._test_indices) == 7
        assert (list(set(our_data_0._train_indices + our_data_0._test_indices))) == list(range(10))
    except:
        print("There are errors that likely come from split_set")
        errors = True

    # Make sure prime_data sets up the deques correctly, whether
    # sequential or random.
    try:
        our_data_0.prime_data(order=NNData.Order.SEQUENTIAL)
        assert len(our_data_0._train_pool) == 3
        assert len(our_data_0._test_pool) == 7
        assert our_data_0._train_indices == list(our_data_0._train_pool)
        assert our_data_0._test_indices == list(our_data_0._test_pool)
        our_big_data.prime_data(order=NNData.Order.RANDOM)
        assert our_big_data._train_indices != list(our_big_data._train_pool)
        assert our_big_data._test_indices != list(our_big_data._test_pool)
    except:
        print("There are errors that likely come from prime_data")
        errors = True

    # Make sure get_one_item is returning the correct values, and
    # that pool_is_empty functions correctly.
    try:
        our_data_1.prime_data(order=NNData.Order.SEQUENTIAL)
        my_x_list = []
        my_y_list = []
        while not our_data_1.pool_is_empty():
            example = our_data_1.get_one_item()
            my_x_list.append(list(example[0]))
            my_y_list.append(list(example[1]))
        assert len(my_x_list) == 2
        assert my_x_list != my_y_list
        my_matched_x_list = [i[0] * 10 for i in my_y_list]
        assert my_matched_x_list == my_x_list
        while not our_data_1.pool_is_empty(our_data_1.Set.TEST):
            example = our_data_1.get_one_item(our_data_1.Set.TEST)
            my_x_list.append(list(example[0]))
            my_y_list.append(list(example[1]))
        assert my_x_list != my_y_list
        my_matched_x_list = [i[0] * 10 for i in my_y_list]
        assert my_matched_x_list == my_x_list
        assert set(i[0] for i in my_x_list) == set(i[0] for i in x)
        assert set(i[0] for i in my_y_list) == set(i[0] for i in y)
    except:
        print("There are errors that may come from prime_data, but could be from another method")
        errors = True

    # Summary
    if errors:
        print("You have one or more errors.  Please fix them before submitting")
    else:
        print("No errors were identified by the unit test.")
        print("You should still double check that your code meets spec.")
        print("You should also check that PyCharm does not identify any PEP-8 issues.")
"""

# First Unit Test-Assignment 3 Checkpoint 1
"""
def check_point_one_test():

    # Mock up a network with three inputs and three outputs

    inputs = [Neurode(LayerType.INPUT) for _ in range(3)]
    outputs = [Neurode(LayerType.OUTPUT, .01) for _ in range(3)]
    if not inputs[0]._reference_value[MultiLinkNode.Side.DOWNSTREAM] == 0:
        print("Fail - Initial reference value is not zero")
    for node in inputs:
        node.reset_neighbors(outputs, MultiLinkNode.Side.DOWNSTREAM)
    for node in outputs:
        node.reset_neighbors(inputs, MultiLinkNode.Side.UPSTREAM)
    if not inputs[0]._reference_value[MultiLinkNode.Side.DOWNSTREAM] == 7:
        print(inputs[0]._reference_value[MultiLinkNode.Side.DOWNSTREAM])
        print("Fail - Final reference value is not correct")
    if not inputs[0]._reference_value[MultiLinkNode.Side.UPSTREAM] == 0:
        print("Fail - Final reference value is not correct")

    # Report data ready from each input and make sure _check_in
    # only reports True when all nodes have reported

    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 0:
        print("Fail - Initial reporting value is not zero")
    if outputs[0]._check_in(inputs[0], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 1:
        print("Fail - reporting value is not correct")
    if outputs[0]._check_in(inputs[2], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 5:
        print("Fail - reporting value is not correct")
    if outputs[0]._check_in(inputs[2], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in (double fire)")
    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 5:
        print("Fail - reporting value is not correct")
    if not outputs[0]._check_in(inputs[1], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned False after all nodes were"
              "checked in")

    # Report data ready from each output and make sure _check_in
    # only reports True when all nodes have reported

    if inputs[1]._check_in(outputs[0], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if inputs[1]._check_in(outputs[2], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if inputs[1]._check_in(outputs[0], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in (double fire)")
    if not inputs[1]._check_in(outputs[1], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned False after all nodes were"
              "checked in")

    # Check that learning rates were set correctly

    if not inputs[0].learning_rate == .05:
        print("Fail - default learning rate was not set")
    if not outputs[0].learning_rate == .01:
        print("Fail - specified learning rate was not set")

    # Check that weights appear random

    weight_list = list()
    for node in outputs:
        for t_node in inputs:
            if node.get_weight(t_node) in weight_list:
                print("Fail - weights do not appear to be set up properly")
            weight_list.append(node.get_weight(t_node))

"""

# Second Unit Test-Assignment 3 Checkpoint 2
"""
def check_point_two_test():
    inodes = []
    hnodes = []
    onodes = []
    for k in range(2):
        inodes.append(FFNeurode(LayerType.INPUT))
    for k in range(2):
        hnodes.append(FFNeurode(LayerType.HIDDEN))
    onodes.append(FFNeurode(LayerType.OUTPUT))
    for node in inodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in hnodes:
        node.reset_neighbors(inodes, MultiLinkNode.Side.UPSTREAM)
        node.reset_neighbors(onodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in onodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.UPSTREAM)
    try:
        inodes[1].set_input(1)
        assert onodes[0].value == 0
    except:
        print("Error: Neurodes may be firing before receiving all input")
    inodes[0].set_input(0)

    # Since input node 0 has value of 0 and input node 1 has value of
    # one, the value of the hidden layers should be the sigmoid of the
    # weight out of input node 1.

    value_0 = (1 / (1 + np.exp(-hnodes[0]._weights[inodes[1]])))
    value_1 = (1 / (1 + np.exp(-hnodes[1]._weights[inodes[1]])))
    inter = onodes[0]._weights[hnodes[0]] * value_0 + \
            onodes[0]._weights[hnodes[1]] * value_1
    final = (1 / (1 + np.exp(-inter)))
    try:
        print(final, onodes[0].value)
        assert final == onodes[0].value
        assert 0 < final < 1
    except:
        print("Error: Calculation of neurode value may be incorrect")
"""

# Third Unit Test-Assignment 3
"""
def main():
 try:
        test_neurode = BPNeurode(LayerType.HIDDEN)
    except:
        print("Error - Cannot instaniate a BPNeurode object")
        return
    print("Testing Sigmoid Derivative")
    try:
        assert BPNeurode._sigmoid_derivative(0) == 0
        if test_neurode._sigmoid_derivative(.4) == .24:
            print("Pass")
        else:
            print("_sigmoid_derivative is not returning the correct "
                  "result")
    except:
        print("Error - Is _sigmoid_derivative named correctly, created "
              "in BPNeurode and decorated as a static method?")
    print("Testing Instance objects")
    try:
        test_neurode.learning_rate
        test_neurode.delta
        print("Pass")
    except:
        print("Error - Are all instance objects created in __init__()?")

    inodes = []
    hnodes = []
    onodes = []
    for k in range(2):
        inodes.append(FFBPNeurode(LayerType.INPUT))
        hnodes.append(FFBPNeurode(LayerType.HIDDEN))
        onodes.append(FFBPNeurode(LayerType.OUTPUT))
    for node in inodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in hnodes:
        node.reset_neighbors(inodes, MultiLinkNode.Side.UPSTREAM)
        node.reset_neighbors(onodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in onodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.UPSTREAM)
    print("testing learning rate values")
    for node in hnodes:
        print(f"my learning rate is {node.learning_rate}")
    print("Testing check-in")
    try:
        hnodes[0]._reporting_nodes[MultiLinkNode.Side.DOWNSTREAM] = 1
        if hnodes[0]._check_in(onodes[1], MultiLinkNode.Side.DOWNSTREAM) and \
                not hnodes[1]._check_in(onodes[1],
                                        MultiLinkNode.Side.DOWNSTREAM):
            print("Pass")
        else:
            print("Error - _check_in is not responding correctly")
    except:
        print("Error - _check_in is raising an error.  Is it named correctly? "
              "Check your syntax")
    print("Testing calculate_delta on output nodes")
    try:
        onodes[0]._value = .2
        onodes[0]._calculate_delta(.5)
        if .0479 < onodes[0].delta < .0481:
            print("Pass")
        else:
            print("Error - calculate delta is not returning the correct value."
                  "Check the math.")
            print("        Hint: do you have a separate process for hidden "
                  "nodes vs output nodes?")
    except:
        print("Error - calculate_delta is raising an error.  Is it named "
              "correctly?  Check your syntax")
    print("Testing calculate_delta on hidden nodes")
    try:
        onodes[0]._delta = .2
        onodes[1]._delta = .1
        onodes[0]._weights[hnodes[0]] = .4
        onodes[1]._weights[hnodes[0]] = .6
        hnodes[0]._value = .3
        hnodes[0]._calculate_delta()
        if .02939 < hnodes[0].delta < .02941:
            print("Pass")
        else:
            print("Error - calculate delta is not returning the correct value.  "
                  "Check the math.")
            print("        Hint: do you have a separate process for hidden "
                  "nodes vs output nodes?")
    except:
        print("Error - calculate_delta is raising an error.  Is it named correctly?  Check your syntax")
    try:
        print("Testing update_weights")
        hnodes[0]._update_weights()
        if onodes[0].learning_rate == .05:
            if .4 + .06 * onodes[0].learning_rate - .001 < \
                    onodes[0]._weights[hnodes[0]] < \
                    .4 + .06 * onodes[0].learning_rate + .001:
                print("Pass")
            else:
                print("Error - weights not updated correctly.  "
                      "If all other methods passed, check update_weights")
        else:
            print("Error - Learning rate should be .05, please verify")
    except:
        print("Error - update_weights is raising an error.  Is it named "
              "correctly?  Check your syntax")
    print("All that looks good.  Trying to train a trivial dataset "
          "on our network")
    inodes = []
    hnodes = []
    onodes = []
    for k in range(2):
        inodes.append(FFBPNeurode(LayerType.INPUT))
        hnodes.append(FFBPNeurode(LayerType.HIDDEN))
        onodes.append(FFBPNeurode(LayerType.OUTPUT))
    for node in inodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in hnodes:
        node.reset_neighbors(inodes, MultiLinkNode.Side.UPSTREAM)
        node.reset_neighbors(onodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in onodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.UPSTREAM)
    inodes[0].set_input(1)
    inodes[1].set_input(0)
    value1 = onodes[0].value
    value2 = onodes[1].value
    onodes[0].set_expected(0)
    onodes[1].set_expected(1)
    inodes[0].set_input(1)
    inodes[1].set_input(0)
    value1a = onodes[0].value
    value2a = onodes[1].value
    if (value1 - value1a > 0) and (value2a - value2 > 0):
        print("Pass - Learning was done!")
    else:
        print("Fail - the network did not make progress.")
        print("If you hit a wall, be sure to seek help in the discussion "
              "forum, from the instructor and from the tutors")
"""

# First Unit Test-Assignment 4 Checkpoint 1
"""
def dll_test():
    my_list = DoublyLinkedList()
    try:
        my_list.get_current_data()
    except DoublyLinkedList.EmptyListError:
        print("Pass")
    else:
        print("Fail")
    for a in range(3):
        my_list.add_to_head(a)
    if my_list.get_current_data() != 2:
        print("Error")
    my_list.move_forward()
    if my_list.get_current_data() != 1:
        print("Fail")
    my_list.move_forward()
    try:
        my_list.move_forward()
    except IndexError:
        print("Pass")
    else:
        print("Fail")
    if my_list.get_current_data() != 0:
        print("Fail")
    my_list.move_back()
    my_list.remove_after_curr()
    if my_list.get_current_data() != 1:
        print("Fail")
    my_list.move_back()
    if my_list.get_current_data() != 2:
        print("Fail")
    try:
        my_list.move_back()
    except IndexError:
        print("Pass")
    else:
        print("Fail")
    my_list.move_forward()
    if my_list.get_current_data() != 1:
        print("Fail")
"""

# Second Unit Test-Assignment 4
"""
def layer_list_test():
    # create a LayerList with two inputs and four outputs
    my_list = LayerList(2, 4)
    # get a list of the input and output nodes, and make sure we have the right number
    inputs = my_list.input_nodes
    outputs = my_list.output_nodes
    assert len(inputs) == 2
    assert len(outputs) == 4
    # check that each has the right number of connections
    for node in inputs:
        assert len(node._neighbors[MultiLinkNode.Side.DOWNSTREAM]) == 4
    for node in outputs:
        assert len(node._neighbors[MultiLinkNode.Side.UPSTREAM]) == 2
    # check that the connections go to the right place
    for node in inputs:
        out_set = set(node._neighbors[MultiLinkNode.Side.DOWNSTREAM])
        check_set = set(outputs)
        assert out_set == check_set
    for node in outputs:
        in_set = set(node._neighbors[MultiLinkNode.Side.UPSTREAM])
        check_set = set(inputs)
        assert in_set == check_set
    # add a couple layers and check that they arrived in the right order, and that iterate and rev_iterate work
    my_list.reset_to_head()
    my_list.add_layer(3)
    my_list.add_layer(6)
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 6
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 3
    # save this layer to make sure it gets properly removed later
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.OUTPUT
    assert len(my_list.get_current_data()) == 4
    my_list.move_back()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 3
    # check that information flows through all layers
    save_vals = []
    for node in outputs:
        save_vals.append(node.value)
    for node in inputs:
        node.set_input(1)
    for i, node in enumerate(outputs):
        assert save_vals[i] != node.value
    # check that information flows back as well
    save_vals = []
    for node in inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
        save_vals.append(node.delta)
    for node in outputs:
        node.set_expected(1)
    for i, node in enumerate(inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]):
        assert save_vals[i] != node.delta
    # try to remove an output layer
    try:
        my_list.remove_layer()
        assert False
    except IndexError:
        pass
    except:
        assert False
    # move and remove a hidden layer
    save_list = my_list.get_current_data()
    my_list.move_back()
    my_list.remove_layer()
    # check the order of layers again
    my_list.reset_to_head()
    assert my_list.get_current_data()[0].node_type == LayerType.INPUT
    assert len(my_list.get_current_data()) == 2
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 6
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == LayerType.OUTPUT
    assert len(my_list.get_current_data()) == 4
    my_list.move_back()
    assert my_list.get_current_data()[0].node_type == LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 6
    my_list.move_back()
    assert my_list.get_current_data()[0].node_type == LayerType.INPUT
    assert len(my_list.get_current_data()) == 2
    # save a value from the removed layer to make sure it doesn't get changed
    saved_val = save_list[0].value
    # check that information still flows through all layers
    save_vals = []
    for node in outputs:
        save_vals.append(node.value)
    for node in inputs:
        node.set_input(1)
    for i, node in enumerate(outputs):
        assert save_vals[i] != node.value
    # check that information still flows back as well
    save_vals = []
    for node in inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
        save_vals.append(node.delta)
    for node in outputs:
        node.set_expected(1)
    for i, node in enumerate(inputs[1]._neighbors[MultiLinkNode.Side.DOWNSTREAM]):
        assert save_vals[i] != node.delta
    assert saved_val == save_list[0].value
"""

