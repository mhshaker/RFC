import numpy as np


class Venn_calib():

    def fit(self, X, y):
        
        preds = np.argmax(X, axis=1)
        classes = np.unique(preds)
        n_samples = len(preds)
        classes_map = {}

        # create a map
        # keys are classes
        # values ore lists of (prediction,label) pairs
        for c in classes:
            classes_map[c] = []

        for i in range(n_samples):
            classes_map[preds[i]].append((preds[i],y[i]))

        lower = []
        upper = []
        class_num = 0
        # iterate on pairs of each class and compute lower and upper for it
        for pairs in classes_map.values():

            labels_count = {}
            
            for c in classes:
                labels_count[c] = 0 

            for pair in pairs:
                labels_count[pair[1]] += 1

            for c in classes:
                if c == class_num:

                    low = labels_count[c] / (len(pairs)+1)
                    up = (labels_count[c]+1) / (len(pairs)+1)
                    lower.append(low)
                    upper.append(up) 

            class_num += 1

        self.l = np.asarray(lower)
        self.u = np.asarray(upper)

        return self
    
    
    def predict(self, X):
        
        preds = np.argmax(X, axis=1)
        preds = np.array(preds, dtype=float)
        n_samples = len(preds)

        classes = np.unique(preds)
        classes = np.array(classes, dtype=int)

        probs = (self.l + self.u) / 2

        pred_probs = []

        for i in range(n_samples):
            if preds[i] == 0:
                preds[i] = pred_probs.append([probs[0], 1-probs[0]])

            elif preds[i] == 1:
                preds[i] = pred_probs.append([1-probs[1], probs[1]])
        
        return np.asarray(pred_probs)