"""
Created by Dannier Li (Chlerry) between Mar 30 and June 25 in 2020 
"""
import coarse.train, coarse.test

import prediction.train, prediction.inference
import prediction.b1_train, prediction.b1_inference
import prediction.b23_train, prediction.b23_inference
import prediction.train_gnet5_new

import residue.train, residue.inference
import residue.b1_train, residue.b23_train
import residue.b_inference

if __name__ == "__main__":  
################## Original Method: train #################
    # Train compression net
    coarse.train.main()
    
    # Train prediction net
    prediction.train.main()
    
    # Train residue net
    residue.train.main()
    
################## Original Method: test ##################
    # Test compression net
    coarse.test.main()
    
    # Test prediction net
    prediction.inference.main()
    
    # Test residue net
    residue.inference.main()
    
##################### New Method: train ###################
    # Train b1's prediction net
    prediction.b1_train.main()
    
    # Train b1's residue net
    residue.b1_train.main()
    
    # Train b2 and b3's prediction net
    prediction.b23_train.main()
    
    # Train b2 and b3's residue net
    residue.b23_train.main()

##################### New Method: test ####################
    # Test b1's prediction net
    prediction.b1_inference.main()
    
    # Test b2 and b3's prediction net
    prediction.b23_inference.main()
    
    # Test b1, b2 and b3's residue net
    residue.b_inference.main()

#################### GNet 5: Train & Test ##################
    # Train b and b1's prediction gnet 5  
    prediction.train_gnet5_new.main()  

    # Test b's GNet5 prediction
    """
    I would strong recommend using command line to run the code instead of the following. Anyway, if you are still using Spider:
    First, please update prediction.inference.main(): 
        -- regrouped_prediction = predict(decoded, b, bm, testing_ratio)
        ++ regrouped_prediction = predict(decoded, b, bm, testing_ratio, 'prediction_gnet5')
    Then uncomment the following line
    """
    prediction.inference.main()

    # Test b1's GNet5 prediction
    """
    First, please update prediction.b1_inference.main(): 
        -- predicted_b1_frame = pred_inference_b1(decoded, b, bm, training_ratio)
        ++ predicted_b1_frame = pred_inference_b1(decoded, b, bm, training_ratio, 'prediction_gnet5')
    Then uncomment the following line
    """
    prediction.b1_inference.main()

    