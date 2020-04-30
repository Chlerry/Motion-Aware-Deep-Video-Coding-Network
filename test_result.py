def coarse_results(coarse_frames, images, foldername,  test_start, test_end, b):
    
    print("RESULTS")
    width,height = images.shape[1],images.shape[2]
    N_mblocks = (width*height)/(b*b)
    N_test = test_end-test_start
    final = []
    j = test_start
    f = coarse_frames.reshape(-1, int(N_mblocks), int(b*b), 3)
    for n in f: # loop over test frames
        result = np.zeros((width, height, 3))
        i = 0
        for y in range(0, result.shape[0], b):
           for x in range(0, result.shape[1], b):
               res = n[i].reshape(b,b,3)
               result[y:y + b, x:x + b] = res
               i = i + 1
        filename = foldername +str(j)+'.png'
        im_rgb = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im_rgb)
        im.save(filename)
        j = j + 1
        final.append(result)
    
    final = np.array(final)  
    
    pixel_max = 255.0
    mse = []
    psnr = []
    ssims = []
    
    for i in range(N_test): 
        img = np.array(images[i].reshape(width*height, 3), dtype = float)
        res = final[i].reshape(width*height, 3)

        m = mean_squared_error(img, res)
        p = 20 * math.log10( pixel_max / math.sqrt( m ))
        s = ssim(img, res, multichannel=True)
        psnr.append(p)
        ssims.append(s)
        mse.append(m)
    
    amse = np.mean(mse)
    apsnr = np.mean(psnr)
    assim = np.mean(ssims)
    return amse, apsnr, assim