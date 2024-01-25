# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:08:46 2024

@author: Alexander Swade
"""

def none_to_zero(val):
    'Helper function to convert None to 0 or use value otherwise'
    return 0 if val is None else val

class OrderBook():
    def __init__(self, best_ask_price=200000, best_bid_price=0):
        self.best_ask = best_ask_price
        self.best_bid = best_bid_price
        self.asks = {}
        self.bids = {}
        self.best_ask_volume = 0
        self.best_bid_volume = 0
            
    def _push_sell(self, quant, price):        
        new_quant = quant
        while ((self.best_bid >= price) 
               and (len(self.bids) > 0)
               and (new_quant > 0)
               ):
            # Get best bid quantity and clear at price
            best_quant = self.bids[self.best_bid]
            
            if best_quant > new_quant:
                # Clear order
                self.bids[self.best_bid] -= new_quant
                new_quant = 0
            
            else: 
                # Clear all quant at best_price and update new_quant
                self.bids.pop(self.best_bid)                    
                if best_quant == new_quant:
                    new_quant = 0
                else:
                    new_quant -= best_quant
                
                # Find new best bid
                self.best_bid = sorted(self.bids.keys())[0]
        
        if new_quant > 0:                    
            self.asks[price] = none_to_zero(self.asks.get(price)) + new_quant
            
            # Update best ask
            self._update_best_ask(price)
            
    def _push_buy(self, quant, price):        
        new_quant = quant
        while ((self.best_ask <= price) 
               and (len(self.asks) > 0)
               and (new_quant > 0)
               ):
            # Get best bid quantity and clear at price
            best_quant = self.asks[self.best_ask]
            
            if best_quant > new_quant:
                # Clear order
                self.asks[self.best_ask] -= new_quant
                new_quant = 0
            
            else: 
                # Clear all quant at best_price and update new_quant
                self.asks.pop(self.best_ask)                    
                new_quant -= best_quant
                
                # Find new best ask
                self.best_ask = sorted(self.asks.keys())[0]
        
        if new_quant > 0:                    
            self.bids[price] = none_to_zero(self.bids.get(price)) + new_quant
            self._update_best_bid(price)
    
    def _update_best_ask(self, new_price):
        if new_price < self.best_ask:
            self.best_ask = new_price
            
    def _update_best_bid(self, new_price):
        if new_price > self.best_bid:
            self.best_bid = new_price
    
    def _update_volumes(self):
        self.best_ask_volume = none_to_zero(self.asks.get(self.best_ask))
        self.best_bid_volume= none_to_zero(self.bids.get(self.best_bid))
        
    def process_order(self, order):
        quant = order['quantity']
        price = order['price']
        direc = order['direction']
        
        if direc == 'sell':
            self._push_sell(quant, price)
        elif direc == 'buy':
            self._push_buy(quant, price)
        
        self._update_volumes()




# =============================================================================
# Test only
# =============================================================================
if __name__ == '__main__':
    orders = [{'direction': 'sell', 'quantity':100, 'price': 103},
              {'direction': 'sell', 'quantity':20, 'price': 101},
              {'direction': 'buy', 'quantity':50, 'price': 97},
              {'direction': 'sell', 'quantity':50, 'price': 99},
              {'direction': 'buy', 'quantity':50, 'price': 99},
              {'direction': 'sell', 'quantity':30, 'price': 99},
              {'direction': 'buy', 'quantity':40, 'price': 101}]
    
    book = OrderBook()
    for o in orders:
        book.process_order(o)
    
    
    print(f'Best ask price: {book.best_ask} with volume: {book.best_ask_volume}')
    print(f'Best bid price: {book.best_bid} with volume: {book.best_bid_volume}')
        