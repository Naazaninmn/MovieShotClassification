import argparse
import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--max_iterations', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--validate_every', type=int, default=100)

    parser.add_argument('--output_path', type=str, default='.')
    parser.add_argument('--data_path', type=str, default='Data')

    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--test', action='store_true')
    
    # Build options dict
    opt = vars(parser.parse_args())

    if not opt['cpu']:
        assert torch.cuda.is_available(), 'a CUDA capable device is needed for running this experiment'

    opt['output_path'] = f'{opt["output_path"]}/record'

    return opt