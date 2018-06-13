import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2,1,0'
NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

from common import *
from utility.file import *

from net.rate import *
from net.loss import *

from dataset.audio_dataset import *
from dataset.audio_processing_tf import *


# -------------------------------------------------------------------------------------
from train_resnet3 import *
test_augment = valid_augment



#--------------------------------------------------------------

def do_submit():

    out_dir  = '/root/share/project/kaggle/tensorflow/results/se-resnet3-spect-02'
    initial_checkpoint = \
        '/root/share/project/kaggle/tensorflow/results/se-resnet3-spect-02/checkpoint/00020000_model.pth'




    #output
    csv_file    = out_dir +'/submit/submission.csv'
    memmap_file = out_dir +'/submit/probs.uint8.memmap'


    ## setup -----------------------------
    os.makedirs(out_dir +'/submit', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.submit.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\n')


    ## net ---------------------------------
    log.write('** net setting **\n')

    net = Net(in_shape = (1, -1, -1), num_classes=AUDIO_NUM_CLASSES).cuda()
    net.load_state_dict(torch.load(initial_checkpoint))
    net.eval()

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    log.write('%s\n\n'%(type(net)))


    ## dataset ---------------------------------
    log.write('** dataset setting **\n')
    test_dataset = AudioDataset(
                                'test_158538',  mode='test',
                                 transform = test_augment)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = 16,
                        drop_last   = False,
                        num_workers = 8,
                        pin_memory  = True,
                        collate_fn  = collate)
    test_num  = len(test_loader.dataset)


    ## start submission here ####################################################################
    start = timer()

    norm_probs = np.memmap(memmap_file, dtype='uint8', mode='w+', shape=(test_num, AUDIO_NUM_CLASSES))
    names = np.zeros(test_num, np.int32)

    n = 0
    for tensors, labels, indices in test_loader:
        print('\rpredicting: %10d/%d (%0.0f %%)  %0.2f min'%(n, test_num, 100*n/test_num,
                         (timer() - start) / 60), end='',flush=True)
        time.sleep(0.01)

        # forward
        tensors = Variable(tensors,volatile=True).cuda(async=True)
        logits  = data_parallel(net, tensors)
        probs   = F.softmax(logits,dim=1)
        labels  = probs.topk(1)[1]

        labels = labels.data.cpu().numpy().reshape(-1)
        probs  = probs.data.cpu().numpy()*255
        probs  = probs.astype(np.uint8)

        batch_size = len(indices)
        names[n:n+batch_size]=labels
        norm_probs[n:n+batch_size]=probs
        n += batch_size

    print('\n')
    assert(n == len(test_loader.sampler) and n == test_num)


    ## submission csv  ----------------------------
    fnames = [id.split('/')[-1] for id in test_dataset.ids]
    names  = [AUDIO_NAMES[l] for l in names]
    df = pd.DataFrame({ 'fname' : fnames , 'label' : names})
    df.to_csv(csv_file, index=False)



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    do_submit()


    print('\nsucess!')