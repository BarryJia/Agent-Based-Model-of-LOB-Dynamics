"""Definition of the environment
"""
from email import message
import numpy as np
import random
import math
from sortedcontainers import SortedDict


class MessageBook():
    """Definition of the message book.

    Arguments:
        - dt - time step
        - start time (default corresponding to 9:30)
        - end time (default corresponding to 16:00)
        - outfile (logfile name)
    """

    def __init__(self, dt=0.01, start_time=34200, end_time=57600, outfile='messages.csv') -> None:
        """Initialize parameters and open logfile for writing messages.
        """
        self.dt = dt
        self.start_time = start_time
        self.end_time = end_time
        self.cur_time = start_time
        self.fout = open(outfile, 'w+')

    def step_time(self):
        """Step time in the message book.
        """
        self.cur_time += self.dt

    def receive_message(self, order):
        """Receive message and write to logfile.
        """
        order_type, order_id, volume,  price, direction = order
        msg = [self.cur_time, order_type, order_id, volume, price, direction]
        msg = [str(item) for item in msg]
        self.fout.write(','.join(msg)+'\n')


class LOB_Environment():
    """Definition of the LOB environment with mechanisms.

    Arguments:
        - message_book - MessageBook object for writing order messages
        - starting_lobs - list of two dictionaries (ask and bid respectively) with starting state of LOBs
        - outfile - logbook filename for printing LOB status

    LOB are dictionaries with prices as keys and list of order_ids and volumes as values.
    The dictionaries are sorted by price and lists are sorted by time of order which created volume.

    E.g. ask LOB lists volumes on ask (sell) side
    Example key,value pair
    price: [order_id_0, volume_0, order_id_1, volume_1]
    100:   [     74394,      100,      43847,       50]

    LOB states are printed whenever it changes.
    """

    def __init__(self, message_book, starting_lobs=None, outfile='LOB.csv') -> None:
        """Initialize parameters, LOB state and open output stream
        """
        self.message_book = message_book
        self.ids = 1000  # ID of orders
        if starting_lobs is not None:
            self.ask_LOB = starting_lobs[0]
            self.bid_LOB = starting_lobs[1]
        else:
            self.ask_LOB = SortedDict(
                {1000000: [0, 10000], 100000: [2, 10000]})
            self.bid_LOB = SortedDict(
                {-1000000: [1, 10000], -100000: [3, 10000]})
        self.fout = open(outfile, 'w+')

    def get_order(self, order_type, price, order_id, volume, direction):
        """Function which dispatches orders to correct functions

        Returns list of modified orders:
        E.g. [order_id_0, new_volume_0, order_id_1, new_volume_1, ...]
        """
        if direction not in [-1, 1]:
            raise ValueError("Incorrect direction")
        if order_type not in [1, 2, 3, 4, 5]:
            raise ValueError("Incorrect order type")
        # New LO
        if order_type == 1:
            # Increase order id
            self.ids += 1
            # Put order on the book
            self.put_order(price, order_id, volume, direction)
            # Send order message
            self.message_book.receive_message(
                (1, order_id, volume, price, direction))
            # Print LOB state
            self.print_LOB()
            # Return id of created position and the volume
            return [order_id, volume]
        # Partial cancellation
        elif order_type == 2:
            # Partially cancel LO
            self.cancel_order(price, order_id, volume, direction)
            # Send order message
            modified_orders = self.message_book.receive_message(
                (2, order_id, volume, price, direction))
            # Print LOB state
            self.print_LOB()
            # Return id of modified position and the new volume
            return modified_orders
        # LO deletion
        elif order_type == 3:
            # Delete LO
            self.delete_order(price, order_id, volume, direction)
            # Send order message
            self.message_book.receive_message(
                (3, order_id, volume, price, direction))
            # Print LOB state
            self.print_LOB()
            # Return id of modified position and the new volume
            return [order_id, 0]
        # Execution
        elif order_type == 4:
            # Execute MO
            order_ids = self.execute_order(volume, direction)
            # Return ids of changed LOs (deleted or partially depleteed LO)
            return order_ids
        # Hidden order
        elif order_type == 5:
            # Send order message
            self.message_book.receive_message(
                (5, order_id, volume, price, direction))
            self.print_LOB()
            return []

    def put_order(self, price, order_id, volume, direction):
        """Function for putting new LO on LOB
        """
        # If bid side
        if direction == 1:
            # Check if not better than best ask
            best_ask = self.ask_LOB.peekitem(index=0)[0]
            if price >= best_ask:
                raise ValueError("Bidding price bigger equal best ask")
            # Get list corresponding to position
            self.bid_LOB[price] = self.bid_LOB.get(price, [])
            # Append to the list the new order
            self.bid_LOB[price].append(order_id)
            self.bid_LOB[price].append(volume)
            return
        # If ask side
        elif direction == -1:
            # Check if not better than best bid
            best_bid = self.bid_LOB.peekitem(index=-1)[0]
            if price <= best_bid:
                raise ValueError("Asking price smaller equal best bid")
            # Get list corresponding to position
            self.ask_LOB[price] = self.ask_LOB.get(price, [])
            # Append to the list the new order
            self.ask_LOB[price].append(order_id)
            self.ask_LOB[price].append(volume)
            return

    def cancel_order(self, price, order_id, volume, direction):
        """Function for cancelling order

        Returns list with modified order:
        [order_id, new_volume]
        """
        # If bid side
        if direction == 1:
            # Get the list corresponding to position
            if price in self.bid_LOB:
                level_list = self.bid_LOB[price]
                # Iterate over orders to find the correct order_id
                for i in range(0, len(level_list), 2):
                    if level_list[i] == order_id:
                        # Modify the order
                        level_list[i+1] -= volume
                        # Check for correctness of order
                        if level_list[i+1] <= 0:
                            raise ValueError(
                                f"Invalid partial cancellation order. Final order volume: {level_list[i+1]}.")
                        return [order_id, level_list[i+1]]
                # raise ValueError("Order not found on LOB")
        # If ask side
        elif direction == -1:
            # Get the list corresponding to position
            if price in self.ask_LOB:
                level_list = self.ask_LOB[price]
                # Iterate over orders to find the correct order_id
                for i in range(0, len(level_list), 2):
                    if level_list[i] == order_id:
                        # Modify the order
                        level_list[i+1] -= volume
                        # Check for correctness of order
                        if level_list[i+1] <= 0:
                            raise ValueError(
                                f"Invalid partial cancellation order. Final order volume: {level_list[i+1]}.")
                        return [order_id, level_list[i+1]]
                # raise ValueError("Order not found on LOB")

    def delete_order(self, price, order_id, volume, direction):
        """Function for deleting LOs
        """
        # If bid side
        if direction == 1:
            # Get the list corresponding to position
            if price in self.bid_LOB:
                level_list = self.bid_LOB[price]
                initial_list_length = len(level_list)
                # print(level_list)
                # print(order_id)
                # Iterate over orders to find the correct order_id
                for i in range(0, len(level_list), 2):
                    # Delete order
                    if level_list[i] == order_id:
                        level_list.pop(i)
                        level_list.pop(i)
                        break
                # if len(level_list) == initial_list_length:
                #     print(level_list, order_id)
                #     raise ValueError("Order not found on LOB")
                # Delete price level if no LOs at given price level
                if len(level_list) == 0:
                    self.bid_LOB.pop(price)
        # If ask side
        elif direction == -1:
            # Get the list corresponding to position
            if price in self.ask_LOB:
                level_list = self.ask_LOB[price]
                initial_list_length = len(level_list)
                # Iterate over orders to find the correct order_id
                for i in range(0, len(level_list), 2):
                    # Delete order
                    if level_list[i] == order_id:
                        level_list.pop(i)
                        level_list.pop(i)
                        break
                # if len(level_list) == initial_list_length:
                #     print(level_list, order_id)
                #     print(price)
                #     raise ValueError("Order not found on LOB")
                # Delete price level if no LOs at given price level
                if len(level_list) == 0:
                    self.ask_LOB.pop(price)

    def execute_order(self, volume, direction):
        """Function for executing orders

        Returns list of modified orders:
        E.g. [order_id_0, new_volume_0, order_id_1, new_volume_1]
        """
        # Create list of modified orders
        modified_orders = []
        # If bid side
        if direction == 1:
            # While order not fully executed
            while volume > 0:
                # Get best bid level
                best_bid = self.bid_LOB.peekitem(index=-1)[0]
                # Get list of LOs at best bid
                level_list = self.bid_LOB[best_bid]
                # Get first LO volume on price level
                cur_vol = level_list[1]
                # If LO bigger than remaining MO volume
                if cur_vol > volume:
                    # Modify current LO
                    level_list[1] -= volume
                    # Send order message
                    self.message_book.receive_message(
                        (4, level_list[0], volume, best_bid, direction))
                    # Update modified orders
                    modified_orders.append(level_list[0])
                    modified_orders.append(level_list[1])
                    # Update remaining MO volume
                    volume = 0
                    # Print LOB
                    self.print_LOB()
                else:  # Else if MO volume bigger than LO
                    # Update remaining MO volume
                    volume -= cur_vol
                    # Send order message
                    self.message_book.receive_message(
                        (4, level_list[0], cur_vol, best_bid, direction))
                    # Update modified orders
                    modified_orders.append(level_list[0])
                    modified_orders.append(0)
                    # Update LOB
                    level_list.pop(0)
                    level_list.pop(0)
                    if len(level_list) == 0:
                        self.bid_LOB.pop(best_bid)
                    # Print LOB
                    self.print_LOB()
        # If ask side
        elif direction == -1:
            # While order not fully executed
            while volume > 0:
                # Get best ask level
                best_ask = self.ask_LOB.peekitem(index=0)[0]
                # Get list of LOs at best ask
                level_list = self.ask_LOB[best_ask]
                # Get first LO volume on price level
                cur_vol = level_list[1]
                # If LO bigger than remaining MO volume
                if cur_vol > volume:
                    # Modify current LO
                    level_list[1] -= volume
                    # Send order message
                    self.message_book.receive_message(
                        (4, level_list[0], volume, best_ask, direction))
                    # Update modified orders
                    modified_orders.append(level_list[0])
                    modified_orders.append(level_list[1])
                    # Update remaining MO volume
                    volume = 0
                    # Print LOB
                    self.print_LOB()
                else:  # Else if MO volume bigger than LO
                    # Update remaining MO volume
                    volume -= cur_vol
                    # Send order message
                    self.message_book.receive_message(
                        (4, level_list[0], cur_vol, best_ask, direction))
                    # Update modified orders
                    modified_orders.append(level_list[0])
                    modified_orders.append(0)
                    # Update LOB
                    level_list.pop(0)
                    level_list.pop(0)
                    if len(level_list) == 0:
                        self.ask_LOB.pop(best_ask)
                    # Print LOB
                    self.print_LOB()
        return modified_orders

    def get_LOB(self, no_levels, normalized=False):
        """Function which gets no_levels levels of LOB on both ask and bid sides.

        Arguments:
            - no_levels - number of levels
            - normalized (bool) - if LOB state to be normalized by midprice (default: False)

        Returns:
        [best_ask, best_ask_vol, best_bid, best_bid_vol,
         second_best_ask, second_best_ask_vol, second_best_bid, second_best_bid_vol, ...]
        List length is no_levels * 4.
        """
        # Initialize list with LOB state
        LOB_state = []
        # Get list of best bid and best ask prices
        bid_prices = list(self.bid_LOB.keys())[::-1]
        ask_prices = list(self.ask_LOB.keys())
        # Pad with =-9999999999 if lists are too short
        while len(bid_prices) < no_levels:
            bid_prices.append(-9999999999)
        while len(ask_prices) < no_levels:
            ask_prices.append(9999999999)
        # Normalize if needed
        midprice = 0
        if normalized:
            midprice = (bid_prices[0]+ask_prices[0])/2
        # Construct LOB state list
        for level in range(no_levels):
            # Append ask price
            LOB_state.append(ask_prices[level]-midprice)
            # Get volumes at ask price
            ask_price_list = self.ask_LOB.get(ask_prices[level], [0, 0])
            # Append total volume at ask price to the LOB state list
            LOB_state.append(np.sum(ask_price_list[1::2]))
            # Append bid price
            LOB_state.append(bid_prices[level-midprice])
            # Get volumes at bid price
            bid_price_list = self.bid_LOB.get(bid_prices[level], [0, 0])
            # Append total volume at bid price to the LOB state list
            LOB_state.append(np.sum(bid_price_list[1::2]))
        return LOB_state

    def print_LOB(self, no_levels=10, normalized=False):
        """Function which prints LOB state to the output file

        Arguments:
            - number of levels of LOB for printing
        """
        # Get LOB state
        LOB_state = self.get_LOB(no_levels=no_levels)
        # Change to list of strings
        msg = [str(item) for item in LOB_state]
        self.fout.write(','.join(msg)+'\n')

    def get_best(self):
        """Function which returns best bid and best ask
        """
        # Get best bid and best ask
        best_bid = self.bid_LOB.peekitem(index=-1)[0]
        best_ask = self.ask_LOB.peekitem(index=0)[0]
        return best_bid, best_ask

def random_index(rate):
    # """随机变量的概率函数"""
    # 参数rate为list<int>
    # 返回概率事件的下标索引
    start = 0
    index = 0
    randnum = random.randint(1, sum(rate))
    for index, scope in enumerate(rate):
        start += scope
        if randnum <= start:
            break
    return index

def dict2rate(dic):
    lis = []
    max_value = max(dic.keys())
    for i in range(max_value+1):
        if i in dic:
            lis.append(dic[i])
        else:
            lis.append(0)
    return lis

def TimeWindowCreation():
    times = []
    time_increments = 10 * 60 * 10 ** 9
    # Iterating over 10 minute windows in the day
    for i in range(base_time + time_increments, final_time + 1, time_increments):
        times.append(i)
    return times

def inWhichTimeWindow(time_value, time_list):
    answer = []
    for i in time_list:
        answer.append(abs(time_value - i))
    return answer.index(min(answer))

def PriceFunction(x, a, b):
    return ((a*x)+b)

if __name__ == "__main__":
    mb = MessageBook()
    lob = LOB_Environment(message_book=mb)
    order_id = 0
    agent_dict = {}
    # lob.put_order(100, 1001, 5, 1)
    # get_order(order_type, price, order_id, volume, direction)
    # order_type = {1,2,3,4,5}
    # direction = {-1, 1}, 1 = bid, -1 = ask
    # bid < ask
    company = 'AMZN'
    path = '/Users/jiashichao/Desktop/Edinburgh/Sem1/Data-driven_Business_and_Behaviour_Analytics/ass2/results/inspect_message/' + company + '/'
    # Loading messagedata from csv file
    data = np.loadtxt(
        '/Users/jiashichao/Desktop/Edinburgh/Sem1/Data-driven_Business_and_Behaviour_Analytics/ass2/LOBSTER_Data/LOBSTER_SampleFile_' + company + '_2012-06-21_10/' + company + '_2012-06-21_34200000_57600000_message_10.csv',
        delimiter=',')
    time = (data[:, 0]*10**9).astype(np.int64)
    order_types = (data[:, 1]).astype(int)  # Order type
    order_ids = (data[:, 2]).astype(int)  # Order ID
    volumes = (data[:, 3]).astype(int)  # Volume
    prices = (data[:, 4]//100).astype(int)  # Price
    directions = (data[:, 5]).astype(int)  # Direction
    bid_price_list = []
    ask_price_list = []
    price_set = set(prices)

    # Iterating over 10 minute windows in the day
    base_time = 34200000000000  # Starting time
    time_increments = 10 * 60 * 10 ** 9  # Time increments
    final_time = 57600000000000  # Final time
    times = []
    bid_price = []
    ask_price = []
    bid_volume = []
    ask_volume = []
    it = 0
    for i in range(base_time+time_increments,final_time+1, time_increments):
        times.append(i//10**9)
        # If LOB state fits within time window
        current_bid_price = []
        current_ask_price = []
        current_bid_volume = []
        current_ask_volume = []
        while it<order_types.shape[0] and time[it]<=i:
            # Increment index of LOB
            if directions[it] == 1:
                current_bid_price.append(prices[it])
                current_bid_volume.append(volumes[it])
            else:
                current_ask_price.append(prices[it])
                current_ask_volume.append(volumes[it])
            it += 1
        bid_price.append(current_bid_price)
        ask_price.append(current_ask_price)
        bid_volume.append(current_bid_volume)
        ask_volume.append(current_ask_volume)

    order_type_prob = []
    user_and_price = []
    with open('/Users/jiashichao/Desktop/Edinburgh/Sem1/Data-driven_Business_and_Behaviour_Analytics/ass2/order_type_prob_per_minute.csv') as distribution_file, open('/Users/jiashichao/Desktop/Edinburgh/Sem1/Data-driven_Business_and_Behaviour_Analytics/ass2/price_prob_per_minute.csv') as user_file:
        distribution = distribution_file.readlines()
        user_distribution = user_file.readlines()
        # print(distribution)
        for i in range(len(distribution)):
            if i > 0:
                current_distribution = distribution[i].strip().split(',')
                order_type_prob.append([int(x) for x in current_distribution[1:6]])

        for i in range(len(user_distribution)):
            if i > 0:
                current_distribution = user_distribution[i].strip().split(',')
                user_and_price.append([round(float(x)) for x in current_distribution[1:]])

        distribution_file.close()
    print(order_type_prob)
    MO_dict = {}
    # every time step:
    base_time = 34200
    final_time = 57600
    timeWindow = TimeWindowCreation()
    current_order_id = 0
    bid_order_dict = {}
    ask_order_dict = {}
    bid_total = 0
    ask_total = 0
    for time_step in range(39):
        # current_user = random.randint(0, len(user_dict))
        # 当前时间内，M个bid用户，N个ask用户
        M = user_and_price[time_step][2]
        N = user_and_price[time_step][3]
        # 对于每一个M中bid用户
        for user_i in range(M+N):
            current_order_id -= 1
            # bid or ask?
            current_order_direction = random_index([M, N])
            current_best_bid, current_best_ask = lob.get_best()
            # bid
            if current_order_direction == 0:
                bid_total += 1
                # 随机选取order type
                current_order_type = random_index(order_type_prob[time_step])
                # print(current_order_type)
                # 随机选取price
                # 随机选取volume
                current_order_price = random.choice(bid_price[time_step])
                current_order_volume = random.choice(bid_volume[time_step])
                if current_order_type == 0:
                    if current_order_price < current_best_ask:
                        print('submission-bid')
                        lob.get_order(1, current_order_price, current_order_id, current_order_volume, 1)
                        bid_order_dict[current_order_id] = [1, current_order_price, current_order_id, current_order_volume]
                # elif current_order_type == 1:
                #     if len(bid_order_dict) > 0:
                #         print('cancellation-bid')
                #         Lowest_Bid_list = sorted(bid_order_dict.items(), key=lambda x: x[1][1])[0]
                #         Lowest_Bid_id = Lowest_Bid_list[0]
                #         Lowest_Bid_volume = Lowest_Bid_list[1][3]
                #         Lowest_Bid_price = Lowest_Bid_list[1][1]
                #         lob.get_order(2, Lowest_Bid_price, Lowest_Bid_id, Lowest_Bid_volume//2, 1)
                #         bid_order_dict[Lowest_Bid_id] = [2, Lowest_Bid_price, Lowest_Bid_id,
                #                                             Lowest_Bid_volume//2]
                # elif current_order_type == 2:
                #     if len(bid_order_dict) > 0:
                #         print('deletion-bid')
                #         Lowest_Bid_list = sorted(bid_order_dict.items(), key=lambda x: x[1][1])[0]
                #         Lowest_Bid_id = Lowest_Bid_list[0]
                #         Lowest_Bid_volume = Lowest_Bid_list[1][3]
                #         Lowest_Bid_price = Lowest_Bid_list[1][1]
                #         # print(bid_order_dict[Lowest_Bid_id], Lowest_Bid_id)
                #         lob.get_order(3, Lowest_Bid_price, Lowest_Bid_id, Lowest_Bid_volume, 1)
                #         bid_order_dict.pop(Lowest_Bid_id)
                elif current_order_type == 3:
                    print('visible-bid')
                    for i in bid_order_dict:
                        if bid_order_dict[i][1] == current_order_price:
                            if bid_order_dict[i][3] > current_order_volume:
                                lob.get_order(4, current_order_price, current_order_id, current_order_volume, 1)
                                old_volume = bid_order_dict[i][3]
                                bid_order_dict[i] = [4, current_order_price, current_order_id, old_volume - current_order_volume]

            # ask
            elif current_order_direction == 1:
                ask_total += 1
                # 随机选取order type
                current_order_type = random_index(order_type_prob[time_step])
                # print(current_order_type)
                # 随机选取price
                # 随机选取volume
                current_order_price = random.choice(ask_price[time_step])
                current_order_volume = random.choice(ask_volume[time_step])
                current_best_bid, current_best_ask = lob.get_best()
                if current_order_type == 0:
                    if current_order_price > current_best_bid:
                        print('submission-ask')
                        lob.get_order(1, current_order_price, current_order_id, current_order_volume, -1)
                        ask_order_dict[current_order_id] = [1, current_order_price, current_order_id,
                                                            current_order_volume]
                # elif current_order_type == 1:
                #     if len(bid_order_dict.keys()) > 0:
                #         print('cancellation-ask')
                #         Lowest_Ask_list = sorted(ask_order_dict.items(), key=lambda x: x[1][1], reverse=True)
                #         if len(Lowest_Ask_list)>0:
                #             Lowest_Ask_id = Lowest_Ask_list[0][0]
                #             Lowest_Ask_volume = Lowest_Ask_list[0][1][3]
                #             Lowest_Ask_price = Lowest_Ask_list[0][1][1]
                #             lob.get_order(2, Lowest_Ask_price, Lowest_Ask_id, Lowest_Ask_volume // 2, -1)
                #             ask_order_dict[Lowest_Ask_id] = [2, Lowest_Ask_price, Lowest_Ask_id,
                #                                              Lowest_Ask_volume // 2]
                # elif current_order_type == 2:
                #     if len(ask_order_dict.keys()) > 0:
                #         print('deletion-ask')
                #         Lowest_Ask_list = sorted(ask_order_dict.items(), key=lambda x: x[1][1], reverse=True)[0]
                #         Lowest_Ask_id = Lowest_Ask_list[0]
                #         Lowest_Ask_volume = Lowest_Ask_list[1][3]
                #         Lowest_Ask_price = Lowest_Ask_list[1][1]
                #         print(ask_order_dict[Lowest_Ask_id])
                #         lob.get_order(3, Lowest_Ask_price, Lowest_Ask_id, Lowest_Ask_volume, -1)
                #         ask_order_dict.pop(Lowest_Ask_id)
                elif current_order_type == 3:
                    print('visible-ask')
                    for i in ask_order_dict:
                        if ask_order_dict[i][1] == current_order_price:
                            if ask_order_dict[i][3] > current_order_volume:
                                lob.get_order(4, current_order_price, current_order_id, current_order_volume, -1)
                                old_volume = ask_order_dict[i][3]
                                ask_order_dict[i] = [4, current_order_price, current_order_id,
                                                     old_volume - current_order_volume]
    print(bid_total, ask_total)


