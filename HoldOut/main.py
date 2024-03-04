import logging
import os
from experiment import Experiment
from load_data import build_splits
from parse_args import parse_arguments
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def setup_experiment(opt):

    experiment = Experiment(opt)
    train_loader, validation_loader, test_loader = build_splits()

    return experiment, train_loader, validation_loader, test_loader


def main(opt):
    experiment, train_loader, validation_loader, test_loader = setup_experiment(opt)

    if not opt['test']:  
            
        # Restoring last checkpoint
        if os.path.exists( f'{opt["output_path"]}/last_checkpoint.pth' ):
            iteration, best_accuracy, total_train_loss = experiment.load_checkpoint(
                f'{opt["output_path"]}/last_checkpoint.pth' )
        else:
            logging.info( opt ) 

        # Train loop
        iteration = 0
        best_accuracy = 0
        total_train_loss = 0
        while iteration < opt['max_iterations']:
            for data in train_loader:

                total_train_loss += experiment.train_iteration(data)

                if iteration % opt['print_every'] == 0:
                    logging.info(
                        f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')

                if iteration % opt['validate_every'] == 0:
                    # Running validation
                    val_accuracy, val_loss, _, _, _, _ = experiment.validate( validation_loader )
                    logging.info(
                        f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                    if val_accuracy >= best_accuracy:
                        best_accuracy = val_accuracy
                        experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration,
                                                    best_accuracy, total_train_loss)
                    experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration,
                                                best_accuracy,
                                                total_train_loss)

                iteration += 1
                if iteration > opt['max_iterations']:
                    break
        

    # The best model on the validation set is used on the test set
    # Test
    experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
    test_accuracy, _, test_f1, test_precision, test_recall, test_cm = experiment.validate( test_loader )

    # plotting confusion matrix
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
    logging.info(f'[TEST] Precision best: {(test_precision):.2f}')
    logging.info(f'[TEST] Recall best: {(test_recall):.2f}')


if __name__ == '__main__':
    opt = parse_arguments()

    # Setup output directories
    os.makedirs(opt['output_path'], exist_ok=True)

    # Setup logger
    logging.basicConfig( filename=f'{opt["output_path"]}/log.txt', format='%(message)s', level=logging.INFO,
                         filemode='a' )

    main(opt)
