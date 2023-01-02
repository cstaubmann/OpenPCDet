
def main():
    with open("train-original.txt", mode='r') as f:
        lines = f.readlines()
    with open("train.txt", mode='w') as f:
        f.writelines(lines[::4])


if __name__ == "__main__":
    main()
