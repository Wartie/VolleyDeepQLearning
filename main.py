import sim
import deepq
import pygame

if __name__ == '__main__':
    simEnv = sim.Environment(100000)

    # done = False
    # counter = 0
    # while not done:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True
    #     simEnv.step()
    #     simEnv.clock.tick(30)
    # deepq.learn(simEnv, doubleDQN=True, perform_her=False, simple=False, behavior_preclone=False)
    deepq.evaluate(simEnv)
    # deepq.collect_data(simEnv)
       
