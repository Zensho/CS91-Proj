import numpy as np

def main():
    b = np.load('LR_task_with_antisaccade_synchronised_min_2.npz')
    # print(b.files)
    EEG = b['EEG']
    labels = b['labels']
    # print(np.shape(EEG))
    # print(np.shape(labels))
    # EEG_data = EEG[:4000]
    # labels_name = labels[:4000]
    # np.savez_compressed('testdata_4000.npz', array1=EEG_data, array2=labels_name)
    EEG_data2 = EEG[:2000]
    labels_data2 = labels[:2000]
    # print(np.shape(EEG_data2))
    # print(np.shape(labels_data2))
    np.savez_compressed('LR_task_with_antisaccade_synchronised_min.npz', EEG=EEG_data2, labels=labels_data2)

if __name__=='__main__':
    main()