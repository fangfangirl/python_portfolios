import random
from telegram import InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CommandHandler, CallbackQueryHandler
from name import token

black = '⚫️'
white = '⚪️'
last_say = True

def enc(board):
    # board is a dictionary mapping (row, col) to grid
    # grid = [[board.get((row, col), '') for col in range(8)] for row in range(8)]
    number = 0
    base = 3
    for row in range(8):
        for col in range(8):
            number *= base
            # if grid[row][col] == black:
            if board.get((row, col)) == black:
                number += 2
            # elif grid[row][col] == white:
            elif board.get((row, col)) == white:
                number += 1
    return str(number)


def dec(number):
    board = {}
    base = 3
    for row in [7, 6, 5, 4, 3, 2, 1, 0]:
        for col in [7, 6, 5, 4, 3, 2, 1, 0]:
            if number % 3 == 2:
                board[(row, col)] = black
            elif number % 3 == 1:
                board[(row, col)] = white
            number //= base
    return board

def getwinner(board):
    count_w = 0
    count_b = 0
    for row in range(8):
        for col in range(8):
            if(row, col) in board:
                if board[(row, col)] == black:
                    count_b = count_b +1
                if board[(row, col)] == white:
                    count_w = count_w +1
    if count_b > count_w:
        return 2
    elif count_b < count_w:
        return 1
    else: return 0

def vaild_can(board,color):
    if color == black:
        othercolor = white
    else:
        othercolor = black
    for row in range(8):
        for col in range(8):
            if (row, col) not in board:
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
                for direction in directions:
                    vailded = False
                    dx, dy = direction
                    x, y = row + dx, col + dy
                    while (x, y) in board and board[(x, y)] == othercolor:
                        #print(x, y)
                        x, y = x + dx, y + dy
                        if (x, y) in board and board[(x, y)] == color:
                            vailded = True
                            #print('i i ')
                            return True
    return False


def vaild_can_player(board, row, col, color):
    if color == black:
        othercolor = white
    else:
        othercolor = black
    if (row, col) in board:
        return False
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for direction in directions:
        cann = False
        dx, dy = direction
        x, y = row + dx, col + dy
        while (x, y) in board and board[(x, y)] == othercolor:
            #print('Yes1')
            x, y = x + dx, y + dy
            if (x, y) in board and board[(x, y)] == color:
                return True
    return False

def flip_pieces(board,row,col,color):
    #print('Yes')
    if color == black:
        othercolor = white
    else:
        othercolor = black
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for direction in directions:
        filpped = False
        dx, dy = direction
        x, y = row + dx, col + dy
        while (x, y) in board and board[(x, y)] == othercolor:
            #print('Yes1')
            x, y = x + dx, y + dy
            if (x, y) in board and board[(x, y)] == color:
                filpped = True
                break
        if filpped:
            #print('Yes2')
            x, y = row, col
            x, y = x + dx, y + dy
            while (x, y) in board and board[(x, y)] == othercolor:
                #print('NONO')
                board[(x, y)] = color
                x, y = x + dx, y + dy

def board_markup(board):
    # board will be encoded and embedded to callback_data
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(board.get((row, col), f'{row},{col}'), callback_data=f'{row}{col}{enc(board)}') for col in range(8)]
        for row in range(8)])

def computer(board):
    #print('3333!')
    can = []
    for row in range(8):
        for col in range(8):
            if (row, col) not in board :
                can.append((row, col))
    if can:
        ttt = random.choice(can)
        row = ttt[0]
        col = ttt[1]
        vaildddd = vaild_can_player(board, row, col, white)
        while(vaildddd == False):
            #print('7777?')
            can.remove((row, col))
            ttt = random.choice(can)
            row = ttt[0]
            col = ttt[1]
            vaildddd = vaild_can_player(board, row, col, white)
            #print(row, col)
            #print(f' {vaildddd}7777?')
        #print(row, col)
        board[(row, col)] = white
        flip_pieces(board, row, col, white)
    return board

# Define a few command handlers. These usually take the two arguments update and
# context.
async def func(update, context):

    data = update.callback_query.data
    row = int(data[0])
    col = int(data[1])
    board = dec(int(data[2:]))
    #flag = 0
    can_b = 0
    can_w = 0
    vaild_b = vaild_can_player(board, row, col, black)
    winner = -1
    if vaild_b == True:
        board[(row, col)] = black
        flip_pieces(board, row, col, black)
        await context.bot.edit_message_text('換電腦進行',
                                            reply_markup=board_markup(board),
                                            chat_id=update.callback_query.message.chat_id,
                                            message_id=update.callback_query.message.message_id)

        vaildput_w = vaild_can(board, white)
        #print(f'{vaildput_w} 電腦可以嗎?')
        if vaildput_w == False:
            #print('2222?')
            can_w = 1
            vaildput_b = vaild_can(board, black)
            #print(f'{vaildput_b} 玩家可以嗎?')
            if vaildput_b == True:
                await context.bot.edit_message_text('電腦無法進行，換玩家進行',
                                                    reply_markup=board_markup(board),
                                                    chat_id=update.callback_query.message.chat_id,
                                                    message_id=update.callback_query.message.message_id)
            else:
                winner = getwinner(board)
                if(winner == 2):
                    await context.bot.edit_message_text('玩家獲勝',
                                                        reply_markup=board_markup(board),
                                                        chat_id=update.callback_query.message.chat_id,
                                                        message_id=update.callback_query.message.message_id)
                elif (winner == 1):
                    await context.bot.edit_message_text('電腦獲勝',
                                                        reply_markup=board_markup(board),
                                                        chat_id=update.callback_query.message.chat_id,
                                                        message_id=update.callback_query.message.message_id)
                elif (winner == 0):
                    await context.bot.edit_message_text('平手',
                                                        reply_markup=board_markup(board),
                                                        chat_id=update.callback_query.message.chat_id,
                                                        message_id=update.callback_query.message.message_id)

        elif vaildput_w == True:
            #print('1111?')
            board = computer(board)
            vaildput_b = vaild_can(board, black)
            #print(f'{vaildput_b} 玩家可以嗎?')
            if vaildput_b == True:
                await context.bot.edit_message_text('換玩家進行1',
                                                    reply_markup=board_markup(board),
                                                    chat_id=update.callback_query.message.chat_id,
                                                    message_id=update.callback_query.message.message_id)
            else:
                winner = getwinner(board)
                if (winner == 2):
                    await context.bot.edit_message_text('玩家獲勝',
                                                        reply_markup=board_markup(board),
                                                        chat_id=update.callback_query.message.chat_id,
                                                        message_id=update.callback_query.message.message_id)
                elif (winner == 1):
                    await context.bot.edit_message_text('電腦獲勝',
                                                        reply_markup=board_markup(board),
                                                        chat_id=update.callback_query.message.chat_id,
                                                        message_id=update.callback_query.message.message_id)
                elif (winner == 0):
                    await context.bot.edit_message_text('平手',
                                                        reply_markup=board_markup(board),
                                                        chat_id=update.callback_query.message.chat_id,
                                                        message_id=update.callback_query.message.message_id)

    else:
        await context.bot.edit_message_text('請按照規則進行',
                                            reply_markup=board_markup(board),
                                            chat_id=update.callback_query.message.chat_id,
                                            message_id=update.callback_query.message.message_id)


async def start_game(update, context):
    board = {(3,3): '⚫️', (3,4): '⚪️', (4,3): '⚪️', (4,4): '⚫️'}
    # reply_markup = board_markup(board)
    await update.message.reply_text('目前盤面', reply_markup=board_markup(board))

def main():
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(token).build()
    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start_game", start_game))
    application.add_handler(CallbackQueryHandler(func))
    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()