import tensorflow as tf
import numpy as np

#list out our bandits
#bandit 4 is set to provide a positive reward most often
#Lower the bandit number the more likely we are to get a positive reward. The higher the bandit number will get a positive reward.
bandits = [0.2,0,-0.2,-5]
num_bandits = len(bandits)

def pullBandit(bandit):
    #Get a random number
    result = np.random.randn(1)
    if result > bandit:
        return 1
    else:
        return -1

#init our agent
tf.reset_default_graph()

#Establish the feedforward part of the network
#this does the actual choosing
weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights,0) #choose the weight with the highest value

#the next six lines establish the traning procedure
#We feed the reward and chosen action into the network
#to compute the loss, and use it to update the network
#placeholders for action and reward
reward_holder = tf.placeholder(shape=[1], dtype =tf.float32) #tensor of rank 1 (shape = [1]) Vector( magnitude and direction)
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
responsible_weight = tf.slice(weights, action_holder, [1])

#Extracts a slice from a tensor. This operation extracts a slice of size in this case 1.
#tf.slice(
 #   input_,
  #  begin,
   # size,
    #name=None
#)

#Policy Loss equation = LOG(P) * A
#P = Policy(Learned Weights)
#A = Advantage (How much better an action was compared to the baseline)
#Default Advantage = Reward-0
#The baseline is 0 in this case

loss = -(tf.log(responsible_weight) * reward_holder) # Our Policy Loss Equation
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001) # Instantiating GradientDescentOptimizer with given learning_rate
update = optimizer.minimize(loss) #When loss minimized we generate a gradient update


total_episodes = 1000 # Amount of loops
total_reward = np.zeros(num_bandits) #Return an array of given shap filled with zeros
e = 0.1 #the chance of taking a random action (epsilon)

init = tf.initialize_all_variables()

#launch TF graph

with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        #choose either random action or one from our network
        if np.random.rand(1) < e: #If epsilon probability is chosen, choose a random bandit
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action)#Current highest weight bandit
        #get our reward from picking one of the bandits
        reward = pullBandit(bandits[action])

        #update the network
        _,resp,ww = sess.run([update, responsible_weight, weights], feed_dict = {reward_holder:[reward],
                                                                                 action_holder:[action]})

        total_reward[action] += reward
        if i % 50 == 0:
            print("Running reward for the "
                  + str(num_bandits) + " bandits: " + str(total_reward))
        i+= 1

print("The agent thinks bandit " + str(np.argmax(ww)+1)
      + " is the most promising....")
if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print("... and it was right!")
else:
    print("...and it was wrong!")
            
