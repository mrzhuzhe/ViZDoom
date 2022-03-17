from torchbeast.monobeast import train
from torch import multiprocessing as mp

class conf:
    def __init__(self):
        #self.env =  "PongNoFrameskip-v4"
        #self.env =  "BoxingNoFrameskip-v0"

        self.savedir = "./runs/teacher_kl_test"
        #self.xpid =  "torchbeast-train_test"
        self.xpid = None
        self.use_lstm = False
        self.mode = "train"
        self.disable_checkpoint = None

        # for produce buffer 
        self.num_actors = 2
        #self.num_actors = 2
        # may not relate to speed
        self.batch_size = 16
        #self.batch_size = 4

        self.num_buffers = max(2 * self.num_actors, self.batch_size)
        # may not relate to speed too 
        #self.num_buffers = 64
        #self.num_learner_threads = 2
        # seems in gpu not bottneck
        self.num_learner_threads = 1
        
        # interval between study ?
        self.unroll_length = 32
        self.disable_cuda = None

        #self.entropy_cost = 0.0006
        self.entropy_cost = 0.01
        self.baseline_cost = 0.5 # td lambda
                
        self.upgo_cost = 0.1
        self.lmb = 0.8

        self.discounting = 0.99
        self.reward_clipping = "abs_one"

        #self.learning_rate = 0.00048
        self.learning_rate = 1e-4
        self.alpha = 0.99
        self.momentum = 0
        self.epsilon = 0.01
        self.grad_norm_clipping = 40.0
        self.total_steps = 1e7

        self.render = False
        self.actor_device_str = "cuda:0"
        self.use_tdlamda = True
        self.use_upgo = True
        self.use_teacher = True
        self.teacher_model_path = "/mnt/e28833eb-0c99-4fe2-802a-09fa58d9c9f5/code/ViZDoom/mytest/runs/1.2MstepTdlambda/model.tar"
        self.teacher_kl_cost = 0.05

    """
     --num_actors 45 \
     --total_steps 30000000 \
     --learning_rate 0.0004 \
     --epsilon 0.01 \
     --entropy_cost 0.01 \
     --batch_size 4 \
     --unroll_length 80 \
     --num_buffers 60 \
     --num_threads 4 \
    """
   


#test(flags)
if __name__ == "__main__":
    mp.set_start_method("spawn")
    flags = conf()
    train(flags)