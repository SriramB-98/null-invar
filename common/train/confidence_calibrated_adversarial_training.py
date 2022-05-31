# common/train/confidence_calibrated_adversarial_training.py

import numpy
import torch
import common.torch
import common.summary
import common.numpy
import attacks
from .adversarial_training import *
import lpips
import numpy as np
from tqdm import tqdm


class ConfidenceCalibratedAdversarialTraining(AdversarialTraining):
    """
    Confidence-calibrated adversarial training.
    """

    def __init__(self, model, trainset, testset, optimizer, scheduler, attack, objective, loss, transition, fraction=0.5, augmentation=None, writer=common.summary.SummaryWriter(), cuda=False, use_lpips=False, use_temp=False):
        """
        Constructor.

        :param model: model
        :type model: torch.nn.Module
        :param trainset: training set
        :type trainset: torch.utils.data.DataLoader
        :param testset: test set
        :type testset: torch.utils.data.DataLoader
        :param optimizer: optimizer
        :type optimizer: torch.optim.Optimizer
        :param scheduler: scheduler
        :type scheduler: torch.optim.LRScheduler
        :param attack: attack
        :type attack: attacks.Attack
        :param objective: objective
        :type objective: attacks.Objective
        :param loss: loss
        :type loss: callable
        :param loss: transition
        :type loss: callable
        :param fraction: fraction of adversarial examples per batch
        :type fraction: float
        :param augmentation: augmentation
        :type augmentation: imgaug.augmenters.Sequential
        :param writer: summary writer
        :type writer: torch.utils.tensorboard.SummaryWriter or TensorboardX equivalent
        :param cuda: run on CUDA device
        :type cuda: bool
        """

        assert fraction < 1

        super(ConfidenceCalibratedAdversarialTraining, self).__init__(model, trainset, testset, optimizer, scheduler, attack, objective, fraction, augmentation, writer, cuda)

        self.loss = loss
        """ (callable) Loss. """

        self.transition = transition
        """ (callable) Transition. """

        self.N_class = None
        """ (int) Number of classes. """
        if getattr(self.model, 'N_class', None) is not None:
            self.N_class = self.model.N_class
        self.use_lpips = use_lpips
        self.use_temp = use_temp
        self.writer.add_text('config/loss', self.loss.__name__)
        self.writer.add_text('config/transition', self.transition.__name__)
        
        self.lpips_dist_func = lpips.LPIPS(net='alex')
        self.lpips_dist_func.to(torch.device('cuda' if self.cuda else 'cpu'))
        
    def lpips_dist(self, a, b):

        if a.shape[-1] < 32:
            pad_len = (32 - a.shape[-1])//2 + 1	
        else:
            pad_len = 0
        a = torch.nn.functional.pad(a, (pad_len, pad_len, pad_len, pad_len), value=-1)
        b = torch.nn.functional.pad(b, (pad_len, pad_len, pad_len, pad_len), value=-1)
        return self.lpips_dist_func(a, b)

    def train(self, epoch):
        """
        Training step.

        :param epoch: epoch
        :type epoch: int
        """
        for b, (inputs, targets) in enumerate(tqdm(self.trainset)):
            if self.augmentation is not None:
                inputs = self.augmentation.augment_images(inputs.numpy())

            inputs = common.torch.as_variable(inputs, self.cuda)
            inputs = inputs.permute(0, 3, 1, 2)
            targets = common.torch.as_variable(targets, self.cuda)

            if self.N_class is None:
                _ = self.model.forward(inputs)
                self.N_class = _.size(1)

            distributions = common.torch.one_hot(targets, self.N_class)

            split = int(self.fraction * inputs.size()[0])
            # update fraction for correct loss computation
            fraction = split / float(inputs.size(0))

            clean_inputs = inputs[:split]
            adversarial_inputs = inputs[split:]
            clean_targets = targets[:split]
            adversarial_targets = targets[split:]
            clean_distributions = distributions[:split]
            adversarial_distributions = distributions[split:]

            self.model.eval()
            self.objective.set(adversarial_targets)
              
            if self.use_lpips:
                self.attack.projection.projections[0].epsilon = 4/255 + np.random.rand()*8/255 # change
            
            # self.attack.projection.projections[0].epsilon = 0.03 # change
            
            adversarial_perturbations, adversarial_objectives = self.attack.run(self.model, adversarial_inputs, self.objective)
            adversarial_perturbations = common.torch.as_variable(adversarial_perturbations, self.cuda)
            adversarial_inputs = adversarial_inputs + adversarial_perturbations     

            # print(lpipsdist)
            # print(adversarial_perturbations.abs().max())
            # import torchvision
            # from matplotlib import pyplot as plt
            # grid_img = torchvision.utils.make_grid(adversarial_inputs.detach().cpu(), nrow=5)
            # plt.imshow(grid_img.permute(1, 2, 0))
            # plt.savefig('adv.png')
            # grid_img = torchvision.utils.make_grid((adversarial_inputs-adversarial_perturbations).detach().cpu(), nrow=5)
            # plt.imshow(grid_img.permute(1, 2, 0))
            # plt.savefig('cln.png')
            # exit()
            
            if self.use_lpips:
                lpipsdist = self.lpips_dist(2*(adversarial_inputs-adversarial_perturbations)-1 , 2*adversarial_inputs-1).reshape(-1) # change
                gamma, adversarial_norms = self.transition(adversarial_perturbations, lpipsdist=lpipsdist) # change
            else:
                gamma, adversarial_norms = self.transition(adversarial_perturbations) # change
                
            gamma = common.torch.expand_as(gamma, adversarial_distributions)

            adversarial_distributions = adversarial_distributions*(1 - gamma)
            adversarial_distributions += gamma*torch.ones_like(adversarial_distributions)/self.N_class

            inputs = torch.cat((clean_inputs, adversarial_inputs), dim=0)

            self.model.train()
            self.optimizer.zero_grad()
            if self.use_temp:
                logits, temps = self.model(inputs, get_temps=True)
            else:
                logits = self.model(inputs)
            clean_logits = logits[:split]
            adversarial_logits = logits[split:]
            #print(logits.shape, adversarial_distributions.shape)
            adversarial_loss = self.loss(adversarial_logits, adversarial_distributions)
            adversarial_error = common.torch.classification_error(adversarial_logits, adversarial_targets)

            clean_loss = self.loss(clean_logits, clean_distributions)
            clean_error = common.torch.classification_error(clean_logits, clean_targets)
            loss = (1 - fraction) * clean_loss + fraction * adversarial_loss
            if self.use_temp:
                loss = loss + 0.001*torch.sum((temps - 1)**2)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            global_step = epoch * len(self.trainset) + b

            print('train/lr', self.scheduler.get_lr()[0])

            print('train/loss', clean_loss.item())
            print('train/error', clean_error.item())
            print('train/confidence', torch.mean(torch.max(torch.nn.functional.softmax(clean_logits, dim=1), dim=1)[0]).item())

            success = torch.clamp(torch.abs(adversarial_targets - torch.max(torch.nn.functional.softmax(adversarial_logits, dim=1), dim=1)[1]), max=1)
            print('train/adversarial_loss', adversarial_loss.item())
            print('train/adversarial_error', adversarial_error.item())
            print('train/adversarial_confidence', torch.mean(torch.max(torch.nn.functional.softmax(adversarial_logits, dim=1), dim=1)[0]).item())
            print('train/adversarial_success', torch.mean(success.float()).item())

                        

            self.writer.add_scalar('train/lr', self.scheduler.get_lr()[0], global_step=global_step)

            self.writer.add_scalar('train/loss', clean_loss.item(), global_step=global_step)
            self.writer.add_scalar('train/error', clean_error.item(), global_step=global_step)
            self.writer.add_scalar('train/confidence', torch.mean(torch.max(torch.nn.functional.softmax(clean_logits, dim=1), dim=1)[0]).item(), global_step=global_step)

            self.writer.add_histogram('train/logits', torch.max(clean_logits, dim=1)[0], global_step=global_step)
            self.writer.add_histogram('train/confidences', torch.max(torch.nn.functional.softmax(clean_logits, dim=1), dim=1)[0], global_step=global_step)

            success = torch.clamp(torch.abs(adversarial_targets - torch.max(torch.nn.functional.softmax(adversarial_logits, dim=1), dim=1)[1]), max=1)
            self.writer.add_scalar('train/adversarial_loss', adversarial_loss.item(), global_step=global_step)
            self.writer.add_scalar('train/adversarial_error', adversarial_error.item(), global_step=global_step)
            self.writer.add_scalar('train/adversarial_confidence', torch.mean(torch.max(torch.nn.functional.softmax(adversarial_logits, dim=1), dim=1)[0]).item(), global_step=global_step)
            self.writer.add_scalar('train/adversarial_success', torch.mean(success.float()).item(), global_step=global_step)

            self.writer.add_histogram('train/adversarial_logits', torch.max(adversarial_logits, dim=1)[0], global_step=global_step)
            self.writer.add_histogram('train/adversarial_confidences', torch.max(torch.nn.functional.softmax(adversarial_logits, dim=1), dim=1)[0], global_step=global_step)

            self.writer.add_histogram('train/adversarial_objectives', adversarial_objectives, global_step=global_step)
            self.writer.add_histogram('train/adversarial_norms', adversarial_norms, global_step=global_step)

            if self.summary_gradients:
                for name, parameter in self.model.named_parameters():
                    self.writer.add_histogram('train_weights/%s' % name, parameter.view(-1), global_step=global_step)
                    self.writer.add_histogram('train_gradients/%s' % name, parameter.grad.view(-1), global_step=global_step)

            self.writer.add_images('train/images', inputs[:min(16, split)], global_step=global_step)
            self.writer.add_images('train/adversarial_images', inputs[split:split + 16], global_step=global_step)
            self.progress(epoch, b, len(self.trainset))

    def test(self, epoch):
        """
        Test on adversarial examples.

        :param epoch: epoch
        :type epoch: int
        """

        self.model.eval()
        # self.attack.projection.projections[0].epsilon = 0.03 # change

        # reason to repeat this here: use correct loss for statistics
        losses = None
        errors = None
        logits = None
        confidences = None

        for b, (inputs, targets) in enumerate(self.testset):
            inputs = common.torch.as_variable(inputs, self.cuda)
            inputs = inputs.permute(0, 3, 1, 2)
            targets = common.torch.as_variable(targets, self.cuda)

            if self.N_class is None:
                _ = self.model.forward(inputs)
                self.N_class = _.size(1)

            distributions = common.torch.as_variable(common.torch.one_hot(targets, self.N_class))

            outputs = self.model(inputs)
            losses = common.numpy.concatenate(losses, self.loss(outputs, distributions, reduction='none').detach().cpu().numpy())
            errors = common.numpy.concatenate(errors, common.torch.classification_error(outputs, targets, reduction='none').detach().cpu().numpy())
            logits = common.numpy.concatenate(logits, torch.max(outputs, dim=1)[0].detach().cpu().numpy())
            confidences = common.numpy.concatenate(confidences, torch.max(torch.nn.functional.softmax(outputs, dim=1), dim=1)[0].detach().cpu().numpy())
            self.progress(epoch, b, len(self.testset))

        global_step = epoch  # epoch * len(self.trainset) + len(self.trainset) - 1
        self.writer.add_scalar('test/loss', numpy.mean(losses), global_step=global_step)
        self.writer.add_scalar('test/error', numpy.mean(errors), global_step=global_step)
        self.writer.add_scalar('test/logit', numpy.mean(logits), global_step=global_step)
        self.writer.add_scalar('test/confidence', numpy.mean(confidences), global_step=global_step)
        
        print('test/loss', numpy.mean(losses))
        print('test/error', numpy.mean(errors))
        print('test/logit', numpy.mean(logits))
        print('test/confidence', numpy.mean(confidences))

        self.writer.add_histogram('test/losses', losses, global_step=global_step)
        self.writer.add_histogram('test/errors', errors, global_step=global_step)
        self.writer.add_histogram('test/logits', logits, global_step=global_step)
        self.writer.add_histogram('test/confidences', confidences, global_step=global_step)

        self.model.eval()

        losses = None
        errors = None
        logits = None
        confidences = None
        successes = None
        norms = None
        objectives = None

        for b, (inputs, targets) in enumerate(self.testset):
            if b >= self.max_batches:
                break

            inputs = common.torch.as_variable(inputs, self.cuda)
            inputs = inputs.permute(0, 3, 1, 2)
            clean_inputs = inputs
            targets = common.torch.as_variable(targets, self.cuda)
            distributions = common.torch.as_variable(common.torch.one_hot(targets, self.N_class))

            self.objective.set(targets)
            adversarial_perturbations, adversarial_objectives = self.attack.run(self.model, inputs, self.objective)
            objectives = common.numpy.concatenate(objectives, adversarial_objectives)

            adversarial_perturbations = common.torch.as_variable(adversarial_perturbations, self.cuda)
            inputs = inputs + adversarial_perturbations
            
            if self.use_lpips:
                lpipsdist = self.lpips_dist(2*clean_inputs.expand(-1,3,-1,-1)-1 , 2*inputs.expand(-1, 3,-1,-1)-1).reshape(-1)
                gamma, adversarial_norms = self.transition(adversarial_perturbations, lpipsdist=lpipsdist)
            else:
                gamma, adversarial_norms = self.transition(adversarial_perturbations)
            gamma = common.torch.expand_as(gamma, distributions)
            distributions = distributions * (1 - gamma) + gamma * torch.ones_like(distributions) / self.N_class

            outputs = self.model(inputs)
            losses = common.numpy.concatenate(losses, self.loss(outputs, distributions, reduction='none').detach().cpu().numpy())
            errors = common.numpy.concatenate(errors, common.torch.classification_error(outputs, targets, reduction='none').detach().cpu().numpy())
            logits = common.numpy.concatenate(logits, torch.max(outputs, dim=1)[0].detach().cpu().numpy())
            confidences = common.numpy.concatenate(confidences, torch.max(torch.nn.functional.softmax(outputs, dim=1), dim=1)[0].detach().cpu().numpy())
            successes = common.numpy.concatenate(successes, torch.clamp(torch.abs(targets - torch.max(torch.nn.functional.softmax(outputs, dim=1), dim=1)[1]), max=1).detach().cpu().numpy())
            norms = common.numpy.concatenate(norms, adversarial_norms.detach().cpu().numpy())
            self.progress(epoch, b, self.max_batches)

        global_step = epoch + 1# * len(self.trainset) + len(self.trainset) - 1
        print('test/adversarial_loss', numpy.mean(losses))
        print('test/adversarial_error', numpy.mean(errors))
        print('test/adversarial_logit', numpy.mean(logits))
        print('test/adversarial_confidence', numpy.mean(confidences))
        print('test/adversarial_norm', numpy.mean(norms))
        print('test/adversarial_objective', numpy.mean(objectives))
        print('test/adversarial_success', numpy.mean(successes))

        self.writer.add_scalar('test/adversarial_loss', numpy.mean(losses), global_step=global_step)
        self.writer.add_scalar('test/adversarial_error', numpy.mean(errors), global_step=global_step)
        self.writer.add_scalar('test/adversarial_logit', numpy.mean(logits), global_step=global_step)
        self.writer.add_scalar('test/adversarial_confidence', numpy.mean(confidences), global_step=global_step)
        self.writer.add_scalar('test/adversarial_norm', numpy.mean(norms), global_step=global_step)
        self.writer.add_scalar('test/adversarial_objective', numpy.mean(objectives), global_step=global_step)
        self.writer.add_scalar('test/adversarial_success', numpy.mean(successes), global_step=global_step)

        self.writer.add_histogram('test/adversarial_losses', losses, global_step=global_step)
        self.writer.add_histogram('test/adversarial_errors', errors, global_step=global_step)
        self.writer.add_histogram('test/adversarial_logits', logits, global_step=global_step)
        self.writer.add_histogram('test/adversarial_confidences', confidences, global_step=global_step)
        self.writer.add_histogram('test/adversarial_norms', norms, global_step=global_step)
        self.writer.add_histogram('test/adversarial_objectives', objectives, global_step=global_step)
