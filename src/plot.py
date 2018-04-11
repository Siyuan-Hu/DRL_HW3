import matplotlib.pyplot as plt

index = 0
step = 1000
update = []
reward = []
std = []

file = open("log.txt", "r")
while True:
	line = file.readline()
	if line:
		if line[0] == '-' or (line[0]>='0' and line[0]<='9'):
			data = line.split()
			reward.append(float(data[0]))
			std.append(float(data[1]))
			# print(data)
			# print(reward)
			# print(std)
			update.append(step*index)
			index += 1
	else:
		break

file.close()

# print(reward)

plt.plot(update, reward, c='b', label='a2c 20 step')

plt.legend(loc='best')
plt.ylabel('mean/std reward on 100 episodes')
plt.xlabel('episodes')
plt.errorbar(update, reward, yerr=std, fmt='o')
plt.grid()
plt.show()
