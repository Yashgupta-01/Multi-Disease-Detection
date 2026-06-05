import sys
from collections import deque

# Map card strings to integer values
CARD_VALUES_MAP = {
    'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '10': 10, 'J': 11, 'Q': 12, 'K': 13
}

class Card:
    __slots__ = ['value', 'suit']

    def __init__(self, value, suit):
        self.value = value
        self.suit = suit

    def __repr__(self):
        return f"({self.value}, {self.suit})"

def get_card_value(card_str):
    """Converts card string (e.g., 'K', '2') to integer value."""
    return CARD_VALUES_MAP.get(card_str, int(card_str))

def rearrange_hand(hand_cards, suit_priority_map):
    def sort_key(card):
        suit_priority_value = suit_priority_map[card.suit]
        return (card.value, suit_priority_value)

    hand_cards.sort(key=sort_key)
    return hand_cards

def solve_game(N, p1_initial, p2_initial, suit_priority_list):
    suit_priority_map = {suit: i + 1 for i, suit in enumerate(suit_priority_list)}

    p1_deck = deque(rearrange_hand(p1_initial, suit_priority_map))
    p2_deck = deque(rearrange_hand(p2_initial, suit_priority_map))

    central_hand = []
    turn = 1  
    
    while p1_deck and p2_deck:
        current_player_deck = p1_deck if turn == 1 else p2_deck
        
        if not current_player_deck:
            break

        current_card = current_player_deck.popleft()
        
        if not central_hand:
            central_hand.append(current_card)
            turn = 2 if turn == 1 else 1
        else:
            top_of_hand = central_hand[-1]
            
            if current_card.value == top_of_hand.value and \
               suit_priority_map[current_card.suit] < suit_priority_map[top_of_hand.suit]:
                
                won_cards = central_hand + [current_card]
                central_hand = []
                
                rearranged_won_cards = rearrange_hand(won_cards, suit_priority_map)
                current_player_deck.extend(rearranged_won_cards)
                
            else:
                central_hand.append(current_card)
                turn = 2 if turn == 1 else 1

    if not p1_deck and not p2_deck:
        return "TIE"
    elif not p2_deck:
        return "WINNER"
    else:
        return "LOSER" 

def main():
    try:
        line1 = sys.stdin.readline()
        if not line1:
            return
        N = int(line1.split()[0])
        
        p1_initial = []
        p2_initial = []
        
        for _ in range(N):
            line = sys.stdin.readline().split()
            c1_str, s1_str, c2_str, s2_str = line
            
            c1_val = get_card_value(c1_str)
            s1_val = int(s1_str)
            c2_val = get_card_value(c2_str)
            s2_val = int(s2_str)
            
            p1_initial.append(Card(c1_val, s1_val))
            p2_initial.append(Card(c2_val, s2_val))
        
        line_priority = sys.stdin.readline().split()
        suit_priority_list = [int(s) for s in line_priority]
        
        result = solve_game(N, p1_initial, p2_initial, suit_priority_list)
        print(result)

    except EOFError:
        pass
    except ValueError:
        pass

if __name__ == "__main__":
    main()
