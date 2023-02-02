class Args() :
    def __init__(self, parser=False) -> None:

        # environment
        self.exec_cap = 50
        self.moving_delay = 2000
        self.query_type = 'tpch' # 'alibaba'
        self.executor_data_point = [5, 10, 20, 40, 50, 60, 80, 100]
        self.num_init_dags = 1
        self.tpch_num = 22
        self.tpch_size = ['2g','5g','10g','20g','50g','80g','100g']
        self.job_folder = './spark_env/tpch/'
        self.num_stream_dags = 200
        self.num_stream_dags_grow = 0.2
        self.stream_interval = 1000 # change to 1
        self.new_dag_interval = 1000 # not used anywhere
        self.warmup_delay = 1000 # node warmup delay keep 1000
        self.learn_obj = 'mean' # makespan
        self.reward_scale = 100000.0
        self.cuda = "cuda"
        self.canvas_base = 10
        self.new_dag_interval_noise = 1000 # not used anywhere

        # Agent
        self.lr = 0.001
        self.batch_size = 32
        self.gamma = 0.9
        self.burnin = 1e3
        self.learn_every = 3
        self.sync_every = 1e3
        self.exploration_rate_decay = 0.999992
        print("Parameters for Environment and Agent set successfully.")


args = Args()
print("Arguments function called successfully.")