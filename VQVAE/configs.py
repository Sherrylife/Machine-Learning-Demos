mnist_cfg1 = dict(dataset_type='MNIST',
                  img_shape=(1, 28, 28),
                  dim=32,
                  n_embedding=32,
                  batch_size=256,
                  n_epochs=20,
                  l_w_embedding=1,
                  l_w_commitment=0.25,
                  lr=2e-4,
                  n_epochs_2=50,
                  batch_size_2=256,
                  pixelcnn_n_blocks=15,
                  pixelcnn_dim=128,
                  pixelcnn_linear_dim=32,
                  vqvae_path='./model_mnist.pth',
                  gen_model_path='./gen_model_mnist.pth')

celebahq_cfg1 = dict(dataset_type='CelebAHQ',
                     img_shape=(3, 128, 128),
                     dim=128,
                     n_embedding=64,
                     batch_size=64,
                     n_epochs=30,
                     l_w_embedding=1,
                     l_w_commitment=0.25,
                     lr=2e-4,
                     n_epochs_2=200,
                     batch_size_2=32,
                     pixelcnn_n_blocks=15,
                     pixelcnn_dim=384,
                     pixelcnn_linear_dim=256,
                     vqvae_path='./model_celebahq_1.pth',
                     gen_model_path='./gen_model_celebahq_1.pth')

celebahq_cfg2 = dict(dataset_type='CelebAHQ',
                     img_shape=(3, 128, 128),
                     dim=128,
                     n_embedding=128,
                     batch_size=64,
                     n_epochs=30,
                     l_w_embedding=1,
                     l_w_commitment=0.25,
                     lr=2e-4,
                     n_epochs_2=200,
                     batch_size_2=32,
                     pixelcnn_n_blocks=15,
                     pixelcnn_dim=384,
                     pixelcnn_linear_dim=256,
                     vqvae_path='./model_celebahq_2.pth',
                     gen_model_path='./gen_model_celebahq_2.pth')

celebahq_cfg3 = dict(dataset_type='CelebAHQ',
                     img_shape=(3, 64, 64),
                     dim=128,
                     n_embedding=64,
                     batch_size=64,
                     n_epochs=20,
                     l_w_embedding=1,
                     l_w_commitment=0.25,
                     lr=2e-4,
                     n_epochs_2=200,
                     batch_size_2=32,
                     pixelcnn_n_blocks=15,
                     pixelcnn_dim=384,
                     pixelcnn_linear_dim=256,
                     vqvae_path='./model_celebahq_3.pth',
                     gen_model_path='./gen_model_celebahq_3.pth')

celebahq_cfg4 = dict(dataset_type='CelebAHQ',
                     img_shape=(3, 64, 64),
                     dim=128,
                     n_embedding=32,
                     batch_size=64,
                     n_epochs=20,
                     l_w_embedding=1,
                     l_w_commitment=0.25,
                     lr=2e-4,
                     n_epochs_2=100,
                     batch_size_2=32,
                     pixelcnn_n_blocks=15,
                     pixelcnn_dim=384,
                     pixelcnn_linear_dim=256,
                     vqvae_path='./model_celebahq_4.pth',
                     gen_model_path='./gen_model_celebahq_4.pth')

cfgs = [mnist_cfg1, celebahq_cfg1, celebahq_cfg2, celebahq_cfg3, celebahq_cfg4]


def get_cfg(id: int):
    return cfgs[id]