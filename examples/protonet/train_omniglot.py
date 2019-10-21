"""Train a prototypical network."""
import argparse

import os
import torch
import numpy as np

from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader

from model import PrototypicalNetwork
from utils import get_prototypes, prototypical_loss, get_accuracies


def _evaluate(model, classes_per_task, dataloader, total_episodes, device):
  model.train(False)

  accuracies = []
  episodes_so_far = 0
  for batch in dataloader:
    train_inputs, train_targets = batch['train']
    train_inputs = train_inputs.to(device=device)
    train_targets = train_targets.to(device=device)
    train_embeddings = model(train_inputs)

    test_inputs, test_targets = batch['test']
    test_inputs = test_inputs.to(device=device)
    test_targets = test_targets.to(device=device)
    test_embeddings = model(test_inputs)

    prototypes = get_prototypes(train_embeddings, train_targets,
                                classes_per_task)
    accs = get_accuracies(prototypes, test_embeddings,
                          test_targets).detach().numpy()
    if episodes_so_far + len(accs) < total_episodes:
      accuracies.extend(accs)
      episodes_so_far += len(accs)
    else:
      remaining = total_episodes - episodes_so_far
      accuracies.extend(accs[:remaining])
      break

  model.train(True)

  mean = 100 * np.mean(accuracies)
  std = 100 * np.std(accuracies)
  ci95 = 1.96 * std / np.sqrt(len(accuracies))

  return mean, ci95


def _train(args):  # pylint: disable=too-many-locals,too-many-statements
  # load training set
  dataset = omniglot(
      args.folder,
      shots=args.num_shots,
      ways=args.train_ways,
      shuffle=True,
      test_shots=5,
      meta_train=True,
      download=args.download,
  )
  train = BatchMetaDataLoader(
      dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
  )

  # load validation
  val_dataset = omniglot(
      args.folder,
      shots=args.num_shots,
      ways=args.test_ways,
      shuffle=True,
      test_shots=5,
      meta_val=True,
      download=args.download,
  )
  val = BatchMetaDataLoader(
      val_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      drop_last=False,
  )

  # load test
  test_dataset = omniglot(
      args.folder,
      shots=args.num_shots,
      ways=args.test_ways,
      shuffle=True,
      test_shots=5,
      meta_test=True,
      download=args.download,
  )
  test = BatchMetaDataLoader(
      test_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      drop_last=False,
  )

  model = PrototypicalNetwork(
      1,
      args.embedding_size,
      hidden_size=args.hidden_size,
  )
  model.to(device=args.device)
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  # Training loop
  best_lb = 0
  best_mean = 0
  faults = 0
  best_params = None
  for step, batch in enumerate(train, 1):
    model.zero_grad()

    train_inputs, train_targets = batch['train']
    train_inputs = train_inputs.to(device=args.device)
    train_targets = train_targets.to(device=args.device)
    train_embeddings = model(train_inputs)

    test_inputs, test_targets = batch['test']
    test_inputs = test_inputs.to(device=args.device)
    test_targets = test_targets.to(device=args.device)
    test_embeddings = model(test_inputs)

    prototypes = get_prototypes(train_embeddings, train_targets,
                                dataset.num_classes_per_task)
    loss = prototypical_loss(prototypes, test_embeddings, test_targets)

    loss.backward()
    optimizer.step()

    if step % 10 == 0:
      print(f'Step {step}, loss = {loss.item()}')

    if step % args.val_batches == 0:
      mean, ci95 = _evaluate(model, val_dataset.num_classes_per_task, val,
                             args.val_episodes, args.device)
      print(f'Validation accuraccy = {mean:.2f} ± {ci95:.2f}%')

      # early stopping
      lb = mean - ci95
      if lb > best_lb or (np.isclose(lb, best_lb) and mean > best_mean):
        print('New best')
        best_lb = lb
        best_mean = mean
        best_params = model.state_dict()
        faults = 0
      else:
        faults += 1
        print(f'{faults} faults')
        if faults >= args.patience:
          print('Training finished')
          break

  # evaluate on test set
  print('Testing...')
  model.load_state_dict(best_params)
  mean, ci95 = _evaluate(model, test_dataset.num_classes_per_task, test,
                         args.test_episodes, args.device)
  print(f'Final accuraccy = {mean:.2f} ± {ci95:.2f}%')

  # Save model
  if args.output_folder is not None:
    filename = os.path.join(
        args.output_folder, 'protonet_omniglot_'
        '{0}shot_{1}way.pt'.format(args.num_shots, args.num_ways))
    torch.save(best_params, filename)


def _parse_args():
  parser = argparse.ArgumentParser('Prototypical Networks')

  parser.add_argument('folder',
                      type=str,
                      help='Path to the folder the data is downloaded to.')
  parser.add_argument(
      '--num-shots',
      type=int,
      default=5,
      help='Number of examples per class (k in "k-shot", default: 5).')
  parser.add_argument(
      '--train-ways',
      type=int,
      default=60,
      help='Number of classes per task in training (N in "N-way", default: 60).'
  )
  parser.add_argument(
      '--test-ways',
      type=int,
      default=5,
      help='Number of classes per task in testing (N in "N-way", default: 5).',
  )

  parser.add_argument(
      '--embedding-size',
      type=int,
      default=64,
      help='Dimension of the embedding/latent space (default: 64).')
  parser.add_argument(
      '--hidden-size',
      type=int,
      default=64,
      help='Number of channels for each convolutional layer (default: 64).')

  parser.add_argument(
      '--output-folder',
      type=str,
      default=None,
      help='Path to the output folder for saving the model (optional).')
  parser.add_argument('--batch-size',
                      type=int,
                      default=1,
                      help='Number of tasks in a mini-batch of tasks.')
  parser.add_argument('--val-batches',
                      type=int,
                      default=100,
                      help='Number of batches between validations.')
  parser.add_argument('--val-episodes',
                      type=int,
                      default=100,
                      help='Number of validation episodes.')
  parser.add_argument('--test-episodes',
                      type=int,
                      default=10_000,
                      help='Number of test episodes.')
  parser.add_argument('--patience',
                      type=float,
                      default=10,
                      help='Early stopping patience.')
  parser.add_argument('--num-workers',
                      type=int,
                      default=1,
                      help='Number of workers for data loading (default: 1).')
  parser.add_argument('--download',
                      action='store_true',
                      help='Download the Omniglot dataset in the data folder.')
  parser.add_argument('--use-cuda',
                      action='store_true',
                      help='Use CUDA if available.')

  args = parser.parse_args()
  args.device = torch.device(
      'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

  return args


if __name__ == '__main__':
  _train(_parse_args())
