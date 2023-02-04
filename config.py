class Config(object):
    # def __init__(
    #     self,
    #     channels=1,
    #     img_size=28,

    #     batch_size=1,
    #     epochs=5,
    #     lr=1e-3,

    #     beta_start=0.0001,
    #     beta_end=0.02,

    #     timesteps=200,
    # ):
    #     val_dic = locals().copy()
    #     for key, value in val_dic.items():
    #         if key != 'self':
    #             setattr(self, key, value)
    channels=1
    img_size=28

    batch_size=128
    epochs=5
    lr=1e-3

    beta_start=0.0001
    beta_end=0.02

    timesteps=200
    time_dim = 4 * img_size
    
    model_path = 'model.pth'
# config = Config()
# print(config.__dict__)
