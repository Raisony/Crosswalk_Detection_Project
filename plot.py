from matplotlib import *
import matplotlib.pyplot as plt

def plotHistory(history, metrics = [['iou', 'val_iou'], ['iou_thresholded', 'val_iou_thresholded'], ['F1_Score', 'val_F1_Score']], losses=['loss', 'val_loss']):
    
    for metric in metrics:
        plt.figure(figsize=(12,5))
        leg = []
        for m in metric:
            leg.append(m)
            plt.plot(history.history[m], linewidth=3)
            plt.suptitle(m[4:] + ' over epochs')
        plt.legend(leg, loc='best')
        plt.ylabel('metrics')
        plt.xlabel('epoch')
        plt.show()

    plt.figure(figsize=(12,5))    
    for loss in losses:
        plt.plot(history.history[loss], linewidth=3)
    plt.suptitle('loss over epochs')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(losses, loc='best')
    plt.show()
plotHistory(result, metrics = [['iou', 'val_iou'], ['iou_thresholded', 'val_iou_thresholded'], ['F1_score', 'val_F1_score']], losses=['loss', 'val_loss'])
# ['iouCalculate', 'val_iouCalculate'], ['iouThresholdedCalculate', 'val_iouThresholdedCalculate'], ['F1ScoreCalculate', 'val_F1ScoreCalculate']
