import logging
import os
from experiment import Experiment
from load_data import build_splits
from parse_args import parse_arguments
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def setup_experiment(opt):

    experiment = Experiment(opt)
    train_loader, test_loader = build_splits()

    return experiment, train_loader, test_loader


def main(opt):
    experiment, train_loader, test_loader = setup_experiment(opt)

    if not opt['test']:  # Skip training if '--test' flag is set   
            
        # Restore last checkpoint
        if os.path.exists( f'{opt["output_path"]}/last_checkpoint.pth' ):
            iteration, best_accuracy, total_train_loss = experiment.load_checkpoint(
                f'{opt["output_path"]}/last_checkpoint.pth' )
        else:
            logging.info( opt ) 

        # Train loop
        iteration = 0
        best_accuracy = 0
        #total_train_loss = 0
        train_loader_iterator = iter(train_loader)
        
        while iteration < opt['max_iterations']:
            try:
                data = next(train_loader_iterator)
            except StopIteration:
                train_loader_iterator = iter(train_loader)
                data = next(train_loader_iterator)

            #for data in train_loader:

                #total_train_loss += experiment.train_iteration(data)

                # if iteration % opt['print_every'] == 0:
                #     logging.info(
                #         f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')

            if iteration % opt['validate_every'] == 0:
                # Run validation
                train_accuracy = experiment.train_iteration( data )
                logging.info(
                    f'[VAL - {iteration}] Accuracy: {(100 * train_accuracy):.2f}')
                if train_accuracy >= best_accuracy:
                    best_accuracy = train_accuracy
                    experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration,
                                                best_accuracy, total_train_loss)
                experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration,
                                            best_accuracy,
                                            total_train_loss)

                iteration += 1
                if iteration > opt['max_iterations']:
                    break
        

    """
    1)We use the best model(s) on the validation set on the test set
    2)If the experiment is clip_disentangle, we also use 4 next best models
    """
    # Test
    experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
    test_accuracy, _, test_f1, test_cm = experiment.validate( test_loader )

    labels = ['Close Up', 'Medium Close Up', 'Medium Shot', 'Medium Long Shot', 'Long Shot']
    cmd = ConfusionMatrixDisplay(confusion_matrix=test_cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    cmd.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix for Movie Shot Classification')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.savefig("cm.jpg")

    logging.info(f'[TEST] Accuracy best: {(100 * test_accuracy):.2f}')
    logging.info(f'[TEST] F1-score best: {(test_f1):.2f}')


if __name__ == '__main__':
    opt = parse_arguments()

    # Setup output directories
    os.makedirs(opt['output_path'], exist_ok=True)

    # Setup logger
    logging.basicConfig( filename=f'{opt["output_path"]}/log.txt', format='%(message)s', level=logging.INFO,
                         filemode='a' )

    main(opt)
