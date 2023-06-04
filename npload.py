import numpy as np
import xlwt

def npload_point(npload_num):

        #Read numpy data and store it in a txt file
        a = np.load('./outputs/test_3d_' + npload_num + '_output.npy')
        data = a.reshape(1,-1)
        np.savetxt("./outputs/" + npload_num + "_test.txt",data,delimiter=',')
        #Read txt file and write data into excel
        book = xlwt.Workbook(encoding='utf-8',style_compression=0)
        sheet = book.add_sheet(npload_num,cell_overwrite_ok=True)
        col = ('HIP-x','HIP-y','HIP-z','R_HIP-x','R_HIP-y','R_HIP-z','R_KNEE-x','R_KNEE-y','R_KNEE-z','R_FOOT-x','R_FOOT-y','R_FOOT-z','L_HIP-x','L_HIP-y','L_HIP-z','L_KNEE-x','L_KNEE-y','L_KNEE-z','L_FOOT-x','L_FOOT-y','L_FOOT-z','SPINE-x','SPINE-y','SPINE-z','THORAX-x','THORAX-y','THORAX-z','NOSE-x','NOSE-y','NOSE-z','HEAD-x','HEAD-y','HEAD-z','L_SHOULDER-x','L_SHOULDER-y','L_SHOULDER-z','L_ELBOW-x','L_ELBOW-y','L_ELBOW-z','L_WRIST-x','L_WRIST-y','L_WRIST-z','R_SHOULDER-x','R_SHOULDER-y','R_SHOULDER-z','R_ELBOW-x','R_ELBOW-y','R_ELBOW-z','R_WRIST-x','R_WRIST-y','R_WRIST-z')
        for i in range(0,51):
                sheet.write(0,i,col[i])

        file_object = open('./outputs/' + npload_num + '_test.txt')
        try:
                b = file_object.read()
        finally:
                file_object.close()

        #b = str(data)
        c = b.split(',')
        c_len=len(c)
        for i in range(1,c_len+1):
                if i%51 ==0:
                        xline = i//51
                        yline = 50
                else:
                        xline = i//51 + 1
                        yline = i%51 - 1
                sheet.write(xline,yline,c[i-1])

        savepath = './outputs/' + npload_num + '.xls' #The storage path of your joint position time series data
        book.save(savepath)

#npload_point('1')