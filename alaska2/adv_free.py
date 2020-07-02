# # Free Adversarial Training Module
# global_noise_data = torch.zeros([batch_size, 3, crop_size, crop_size]).cuda()
#
#
# clip_eps: 4.0
# fgsm_step: 4.0
# n_repeats: 4
#
#
# def fgsm(gradz, step_size):
#     return step_size*torch.sign(gradz)
#
# def train(train_loader, model, criterion, optimizer, epoch):
#     global global_noise_data
#     mean = torch.Tensor(np.array(mean)[:, np.newaxis, np.newaxis])
#     mean = mean.expand(3, crop_size, crop_size).cuda()
#     std = torch.Tensor(np.array(std)[:, np.newaxis, np.newaxis])
#     std = std.expand(3, crop_size, crop_size).cuda()
#     # Initialize the meters
#     # switch to train mode
#     model.train()
#     for i, (input, target) in enumerate(train_loader):
#         input = input.cuda(non_blocking=True)
#         target = target.cuda(non_blocking=True)
#         for j in range(n_repeats):
#             # Ascend on the global noise
#             noise_batch = global_noise_data[0:input.size(0)].clone().requires_grad_().cuda()
#             in1 = input + noise_batch
#             in1.clamp_(0, 1.0)
#             in1.sub_(mean).div_(std)
#             output = model(in1)
#             loss = criterion(output, target)
#
#             # compute gradient and do SGD step
#             optimizer.zero_grad()
#             loss.backward()
#
#             # Update the noise for the next iteration
#             pert = fgsm(noise_batch.grad, fgsm_step)
#             global_noise_data[0:input.size(0)] += pert.data
#             global_noise_data.clamp_(-clip_eps, clip_eps)
#
#             optimizer.step()
#             # measure elapsed time
from catalyst.dl import SupervisedRunner


class FreeAdvRunner(SupervisedRunner):
    def _run_batch(self, batch):
        self.state.step += self.state.batch_size
        batch = self._batch2device(batch, self.device)
        self.state.input = batch
        self.state.timer.stop("_timers/data_time")

        for r in range(self.n_repeats):
            self._run_event("batch", moment="start")
            self.state.timer.start("_timers/model_time")

            noise_batch = global_noise_data[0 : batch.size(0)].clone().requires_grad_().cuda()
            in1 = batch + noise_batch
            in1.clamp_(0, 1.0)
            in1.sub_(mean).div_(std)

            self.state.output = self.forward(in1)
            self.state.timer.stop("_timers/model_time")
            self.state.timer.stop("_timers/batch_time")
            self._run_event("batch", moment="end")
