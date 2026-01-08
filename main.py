import argparse


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def PQ(num):
    print("I am PQ")
    print(num**2)

def QSGD(num):
    print("I am QSGD")
    print(num)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--iid', type=str, default='iid')
    parse.add_argument('--dataset', type=str, default='CIFAR-10')
    parse.add_argument('--quan', type=str, default="PQ")
    parse.add_argument('--bit', type=int, default=8)


    args = vars(parse.parse_args())
    for k in args:
        print(f"the key is {k}, the value is {args[k]}")
    print(args['dataset'])
    if args['quan'] == "PQ":
        loss = PQ
    elif args['quan'] == "QSGD":
        loss = QSGD
    loss(10)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
