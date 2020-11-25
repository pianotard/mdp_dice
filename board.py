import random
import deprecation
import itertools as it
import pandas as pd

DEBUG = False

class Board:
        
    CELL_LIMIT = 15
    DECK_LIMIT = 5
    PIP_LIMIT = 7
    
    EMPTY = '0'
    COMBO = 'c'
    GROWTH = 'g'
    JOKER = 'j'
    MIMIC = 'm'
    MOON = 'o'
    
    COMBO_COUNT = 1
    COMBO_DPSPC = 10
    
    MOON_BASE_SPD_UP_PP = 0.15
    MOON_ACTIVE_SPD_UP_PP = 0.18
    
    PLACEHOLDER_DICE = 'x'
    PLACEHOLDER_DPS = 0
    
    CACHED_DPS = {}
    
    ALL_DICE = {
        id_no: {
            'name': name,
            'class': class_lvl,
            'mtd': mtd,
            'atk_spd': spd
        } for id_no, name, class_lvl, mtd, spd in [d.values() \
          for d in pd.read_csv('all_dice.txt').to_dict(orient = 'records')]
    }
    
    @classmethod
    def new_board(cls, deck):
        cls.PLACEHOLDER_DPS = sum([cls([cls.EMPTY] * cls.CELL_LIMIT, deck).spawn_dice(d, 0).dps() / len(deck)\
                              for d in deck])
        return cls([cls.EMPTY] * cls.CELL_LIMIT, deck)
        
    @classmethod
    def next_board(cls, cells, deck):
        return cls(cells, deck)
    
    @classmethod
    def parse_state_str(cls, state_str, deck):
        cells = state_str.split(',')
        return cls(cells, deck)
      
    @classmethod
    def adjacent_cells(cls, i):
        adj = []
        if i < 10:
            adj.append(i + 5)
        if i > 4:
            adj.append(i - 5)
        if i % 5 > 0:
            adj.append(i - 1)
        if i % 5 < 4:
            adj.append(i + 1)
        return adj
        
    def __init__(self, cells, deck):
        """
        :param cells: list of 15 str representing cells. n if empty
        :param deck: list of int representing die in deck
        """
        assert len(cells) == Board.CELL_LIMIT, f'Invalid cells provided: {len(cells)}'
        assert len(deck) == Board.DECK_LIMIT, f'Invalid deck length provided: {len(deck)}'
        self._cells = cells
        self._deck = deck
        self.init_spd_up()
        
    def init_spd_up(self):
        self._spd_ups = [1] * Board.CELL_LIMIT
        
        non_moon_indices = [i for i in range(Board.CELL_LIMIT) \
                            if self.dice_at_cell(i) != Board.MOON or self.dice_at_cell(i) != Board.PLACEHOLDER_DICE]
        num_moons = len([i for i in range(Board.CELL_LIMIT) \
                     if self.dice_at_cell(i) == Board.MOON or self.dice_at_cell(i) == Board.PLACEHOLDER_DICE])
        if num_moons == 0:
            return
        for i in non_moon_indices:
            adj = Board.adjacent_cells(i)
            adj_moon_pips = [self.pip_at_cell(j) for j in adj if self.dice_at_cell(j) == Board.MOON]
            adj_x_pips = [self.pip_at_cell(j) / Board.DECK_LIMIT \
                          for j in adj if self.dice_at_cell(j) == Board.PLACEHOLDER_DICE]
            adj_pips = adj_moon_pips + adj_x_pips
            if len(adj_pips) == 0:
                continue
            max_adj_moon_pip = max(adj_pips)
            if num_moons in [3, 5, 7]:
                self._spd_ups[i] = 1 + max_adj_moon_pip * Board.MOON_ACTIVE_SPD_UP_PP
            else:
                self._spd_ups[i] = 1 + max_adj_moon_pip * Board.MOON_BASE_SPD_UP_PP
        
    def __str__(self):
        ret = ''
        for i in range(3):
            ret += str(self._cells[i * 5:(i + 1) * 5]) + '\n'
        return f'{ret[:-1]}\nDPS: {self.dps()}'
    
    def mdp_params(self, depth = 1, breadth = 5):
        boards = [self]
        s_a_r_dict = {}
        for i in range(depth):
            next_boards = []
            for b in boards:
                s_a_r_dict[b.state_str()] = b.next_states()
                next_b = [v for v in s_a_r_dict[b.state_str()].values()]
                next_boards.extend(next_b)
            boards = sorted(next_boards, key = Board.dps)[-breadth:]
        return s_a_r_dict
    
    def dice_dps(self, dice, cell, pip):
        spd_up = self._spd_ups[cell]
        if dice == Board.PLACEHOLDER_DICE:
            return Board.PLACEHOLDER_DPS * pip * spd_up
        if dice == Board.JOKER:
            same_pip_die_unique = pd.Series([self._cells[j] for j in range(Board.CELL_LIMIT) \
                                            if self.pip_at_cell(j) == pip]).unique()
            return max([self.dice_dps(d, cell, pip) for d in self._deck if d != Board.JOKER])
        mtd = Board.ALL_DICE[dice]['mtd']
        spd = Board.ALL_DICE[dice]['atk_spd']
        if dice == Board.COMBO:
            return (mtd + Board.COMBO_COUNT * Board.COMBO_DPSPC) * pip * spd_up / spd
        if dice == Board.GROWTH and pip < 7:
            return Board.PLACEHOLDER_DPS * min(pip + 1, 7) * spd_up
        return mtd * pip * spd_up / spd
    
    def dps(self):
        if self.state_str() not in Board.CACHED_DPS.keys():
            dps = 0
            for i in range(Board.CELL_LIMIT):
                if self._cells[i] == Board.EMPTY:
                    continue
                    
                dice = self._cells[i][:-1]
                pip = int(self._cells[i][-1])
                dps += self.dice_dps(dice, i, pip)
                
            Board.CACHED_DPS[self.state_str()] = dps
            if DEBUG: print(f'Cached DPS for {self.state_str()}')
            return dps
        else:
            return Board.CACHED_DPS[self.state_str()]
    
    def state_str(self):
        ret = ''
        for cell in self._cells:
            ret += str(cell) + ','
        return ret[:-1]
    
    def next_states(self):
        """
        :return:    dict of {action: result_state} using PLACEHOLDER_DICE to represent
                    random spawning of new dice
        """
        
        states = {}
                
        # Possible merges
        for merge in self.possible_merges():
            states[f'm_{merge[0] + 1}_{merge[1] + 1}'] = self.merge_dice(merge[0], merge[1])
            
        # Growths
        growth_cells = [i for i in range(Board.CELL_LIMIT) \
                        if self._cells[i][:-1] == Board.GROWTH and int(self._cells[i][-1]) != Board.PIP_LIMIT]
        for cell in growth_cells:
            new_pip = int(self._cells[cell][-1]) + 1
            states[f'g_{cell + 1}'] = self.remove_dice(cell).spawn_dice(Board.PLACEHOLDER_DICE, cell, pip = new_pip)
        return states
        
    def possible_merges(self):
        """
        :return: tuple representint src -> dest of possible merges
        """
        die_indices = [i for i in range(self.CELL_LIMIT) if self._cells[i] != self.EMPTY]
        perms = list(it.permutations(die_indices, 2))
        return [perm for perm in perms if self.legal_merge(perm[0], perm[1])]
            
    def empty_cells(self):
        return [i for i in range(len(self._cells)) if self._cells[i] == self.EMPTY]
        
    def remove_dice(self, cell):
        assert cell not in self.empty_cells(), f'Non-empty cell provided: {cell}'
        new_cells = self._cells.copy()
        new_cells[cell] = Board.EMPTY
        return Board(new_cells, self._deck)
        
    def spawn_dice(self, dice, spawn_cell, pip = 1):
        empty_cells = self.empty_cells()
        assert dice in self._deck + [Board.PLACEHOLDER_DICE], f'Invalid dice provided: {dice}'
        assert spawn_cell >= 0 and spawn_cell < self.CELL_LIMIT, f'Spawn loc out of range: {spawn_cell}'
        assert pip >= 1 and pip <= 7, f'Invalid pip provided: {pip}'
        assert spawn_cell in empty_cells, f'Spawn loc already has dice: {spawn_cell}'
        new_cells = self._cells.copy()
        new_cells[spawn_cell] = dice + str(pip)
        return Board(new_cells, self._deck)
    
    def mimic_or_joker(self, src, dest):
        mimic = self._cells[src][:-1] == Board.MIMIC or self._cells[dest][:-1] == Board.MIMIC
        joker = self._cells[src][:-1] == Board.JOKER or self._cells[dest][:-1] == Board.JOKER
        return mimic or joker
    
    def legal_merge(self, src, dest):
        if not (src >= 0 and src < Board.CELL_LIMIT and dest >= 0 and dest < Board.CELL_LIMIT):
            if DEBUG: print(f'Merge locs out of range: {src} -> {dest}')
            return False
        if not (self._cells[src] != Board.EMPTY and self._cells[dest] != Board.EMPTY):
            if DEBUG: print(f'Merge locs contain empty cell: {src} -> {dest}')
        if self._cells[src][:-1] != self._cells[dest][:-1]:
            if not (self.mimic_or_joker(src, dest)):
                if DEBUG: print(f'Merge locs contain different die: {src} -> {dest}')
                return False
        if self._cells[src][-1] != self._cells[dest][-1]:
            if DEBUG: print(f'Merge locs contain different pips: {src} -> {dest}')
            return False
        if int(self._cells[src][-1]) == Board.PIP_LIMIT:
            if DEBUG: print(f'Unable to merge max pips: {src} -> {dest}')
            return False
        return True
    
    def merge_dice(self, src, dest, dice = PLACEHOLDER_DICE):
        """
        :param src, dest: ints representing source and destination indices of merge
        :return: merge outcome with placeholder dice in dest
        """
        assert self.legal_merge(src, dest), f'Illegal merge'
        if self._cells[src][:-1] == Board.JOKER:
            dest_dice = self._cells[dest][:-1]
            dest_pip = int(self._cells[dest][-1])
            return self.remove_dice(src).spawn_dice(dest_dice, src, pip = dest_pip)
        else:
            if dice != Board.PLACEHOLDER_DICE and \
                (self._cells[src][:-1] == Board.COMBO or self._cells[src][:-1] == Board.COMBO):
                Board.COMBO_COUNT += 1
            new_pip = int(self._cells[dest][-1]) + 1
            return self.remove_dice(src).remove_dice(dest).spawn_dice(dice, dest, pip = new_pip)
    
    def dice_at_cell(self, cell):
        return self._cells[cell][:-1]
    
    def pip_at_cell(self, cell):
        return int(self._cells[cell][-1])
    
    def __eq__(self, other):
        if isinstance(other, Board):
            return self.state_str() == other.state_str()
        return False
    
    def __hash__(self):
        return self.state_str().__hash__()

    @deprecation.deprecated()
    def smart_empty_cells(self):
        return [i for i in range(len(self._cells)) \
                if self._cells[i] == self.EMPTY and i > self._largest_filled_cell]

    @deprecation.deprecated()
    def v_mirror(self):
        """
        :return: Reflection of board when a mirror is placed vertically. LR reflection
        """
        return Board(self._cells[4::-1] + self._cells[9:4:-1] + self._cells[14:9:-1], self._deck)
    
    @deprecation.deprecated()
    def h_mirror(self):
        """
        :return: Reflection of board when a mirror is placed horizontally. TD reflection
        """
        return Board(self._cells[10:] + self._cells[5:10] + self._cells[:5], self._deck)
    
    @deprecation.deprecated()
    def symmetrical(self):
        return self.v_mirror() == self or self.h_mirror() == self
    
    @classmethod
    @deprecation.deprecated()
    def remove_symmetry(cls, boards):
        remove_identical = list(pd.Series(boards).unique())
        result = [b for b in remove_identical if not b.symmetrical()]
        return result
    
    @classmethod
    @deprecation.deprecated()
    def get_states(cls, deck):
        """
        Deprecated. Takes too long to run
        :return: list of str representation of ALL possible board states from provided deck
        """
        states = {num: None for num in range(1, cls.CELL_LIMIT + 1)}
        print('Initializing 1-dice boards..', end = '')
        lonely_states = []
        for dice in deck:
            for cell in range(cls.CELL_LIMIT):
                for pip in range(1, cls.PIP_LIMIT + 1):
                    lonely_states.append(Board.new_board(deck).spawn_dice(dice, cell, pip))
        states[1] = lonely_states
        
        print('Done')
        for num_die in range(2, cls.CELL_LIMIT + 1):
            print(f'Generating boards with {num_die} die..', end = '')
            curr_states = []
            for prev_state in states[num_die - 1]:
                empty_cells = prev_state.smart_empty_cells()
                for cell in empty_cells:
                    for dice in deck:
                        for pip in range(1, cls.PIP_LIMIT + 1):
                            curr_states.append(prev_state.spawn_dice(dice, cell, pip))
            states[num_die] = curr_states
            print('Done')
        return states