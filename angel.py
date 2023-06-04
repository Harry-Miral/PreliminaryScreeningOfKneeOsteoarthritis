import math
import pandas as pd
import xlrd
import xlwt
def cal_angle(point_a, point_b, point_c):
    a_x, b_x, c_x = point_a[0], point_b[0], point_c[0]  # x-coordinates of points a, b, c
    a_y, b_y, c_y = point_a[1], point_b[1], point_c[1]  # The y coordinates of points a, b, c
    a_z, b_z, c_z = point_a[2], point_b[2], point_c[2]  # z coordinates of points a, b, c

    x1, y1, z1 = (a_x - b_x), (a_y - b_y), (a_z - b_z)
    x2, y2, z2 = (c_x - b_x), (c_y - b_y), (c_z - b_z)

    cos_b = (x1 * x2 + y1 * y2 + z1 * z2) / (
                math.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2) * (math.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)))  # The cosine value of the included angle of the corner point b
    B = math.degrees(math.acos(cos_b))  # Angle value of corner point b
    return B

def cal_angle2(point_a, point_b, point_c,point_d):
    a_x, b_x, c_x, d_x = point_a[0], point_b[0], point_c[0], point_d[0]  # x-coordinates of points a, b, c
    a_y, b_y, c_y, d_y = point_a[1], point_b[1], point_c[1], point_d[1]  # The y coordinates of points a, b, c
    a_z, b_z, c_z, d_z = point_a[2], point_b[2], point_c[2], point_d[2]  # z coordinates of points a, b, c
    x1, y1, z1 = (a_x - b_x), (a_y - b_y), (a_z - b_z)
    x2, y2, z2 = (c_x - d_x), (c_y - d_y), (c_z - d_z)
    cos_bc = (x1 * x2 + y1 * y2 + z1 * z2) / (
                math.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2) * (math.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)))  # The cosine value of the included angle of the corner point b
    BC = math.degrees(math.acos(cos_bc))  # Angle value of corner point b
    return BC


def angel_all(angel_num,path):
    worksheet = xlrd.open_workbook('./outputs/' + angel_num + '.xls')
    sheet_names = worksheet.sheet_names()

    book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    sheet2 = book.add_sheet(angel_num,cell_overwrite_ok=True)
    col = ('HIP_ANGEL','R_HIP_ANGEL','R_KNEE_ANGEL','L_HIP_ANGEL','L_KNEE_ANGEL','SPINE_ANGEL')
    for i in range(0,6):
        sheet2.write(0,i,col[i])
    angel_tem = 0
    aap = [0, 0, 0]
    bbp = [0, 1, 2]
    ccp = [1, 2, 3]
    ddp = [0, 0, 1]

    file_object = open('./outputs/' + angel_num + '_test.txt')
    try:
        b_b = file_object.read()
    finally:
        file_object.close()

    c_c = b_b.split(',')
    c_len = len(c_c)
    range_num = c_len // 51

    for i in range(0,range_num):
        for sheet_name in sheet_names:
            sheet = worksheet.sheet_by_name(sheet_name)
            #The span angle (HIP_ANGEL) is calculated here
            aap[0] = float(sheet.cell_value(i+1,6))#x coordinate of right knee joint
            aap[1] = float(sheet.cell_value(i+1,7))#The y coordinate of the right knee joint
            aap[2] = float(sheet.cell_value(i+1,8))#Z coordinate of right knee joint
            bbp[0] = float(sheet.cell_value(i+1,0))#center x coordinate
            bbp[1] = float(sheet.cell_value(i+1,1))#center y coordinate
            bbp[2] = float(sheet.cell_value(i+1,2))#center z coordinate
            ccp[0] = float(sheet.cell_value(i+1,15))#x-coordinate of left knee joint
            ccp[1] = float(sheet.cell_value(i+1,16))#The y coordinate of the left knee joint
            ccp[2] = float(sheet.cell_value(i+1,17))#Z coordinate of left knee joint
            angel_tem = cal_angle(aap, bbp, ccp)
            sheet2.write(i+1,0,angel_tem)#Calculate angle (degrees)

            #Right hip bend angle (R_HIP_ANGEL) is calculated here
            aap[0] = float(sheet.cell_value(i+1,6))#x coordinate of right knee joint
            aap[1] = float(sheet.cell_value(i+1,7))#The y coordinate of the right knee joint
            aap[2] = float(sheet.cell_value(i+1,8))#Z coordinate of right knee joint
            bbp[0] = float(sheet.cell_value(i+1,3))#right hip x coordinate
            bbp[1] = float(sheet.cell_value(i+1,4))#right hip y coordinate
            bbp[2] = float(sheet.cell_value(i+1,5))#right hip z coordinate
            ccp[0] = float(sheet.cell_value(i+1,21))#spine x coordinate
            ccp[1] = float(sheet.cell_value(i+1,22))#spine y coordinate
            ccp[2] = float(sheet.cell_value(i+1,23))#spine z coordinate
            ddp[0] = float(sheet.cell_value(i+1,0))  # center x coordinate
            ddp[1] = float(sheet.cell_value(i+1,1))  # center y coordinate
            ddp[2] = float(sheet.cell_value(i+1,2))  # center z coordinate
            angel_tem=cal_angle2(aap, bbp, ccp,ddp)
            sheet2.write(i+1,1,angel_tem)#Calculate angle (degrees)

            #Calculate the left hip bend angle here (L_HIP_ANGEL)
            aap[0] = float(sheet.cell_value(i+1,15))#x-coordinate of left knee joint
            aap[1] = float(sheet.cell_value(i+1,16))#The y coordinate of the left knee joint
            aap[2] = float(sheet.cell_value(i+1,17))#Z coordinate of left knee joint
            bbp[0] = float(sheet.cell_value(i+1,12))#left hip x coordinate
            bbp[1] = float(sheet.cell_value(i+1,13))#left hip y coordinate
            bbp[2] = float(sheet.cell_value(i+1,14))#left hip z coordinate
            angel_tem=cal_angle2(aap, bbp, ccp,ddp)
            sheet2.write(i+1,3,angel_tem)#Calculate angle (degrees)

            #Right knee angle is calculated here (R_KNEE__ANGEL)
            aap[0] = float(sheet.cell_value(i+1,3))#right hip x coordinate
            aap[1] = float(sheet.cell_value(i+1,4))#right hip y coordinate
            aap[2] = float(sheet.cell_value(i+1,5))#right hip z coordinate
            bbp[0] = float(sheet.cell_value(i+1,6))#right knee x coordinate
            bbp[1] = float(sheet.cell_value(i+1,7))#right knee y coordinate
            bbp[2] = float(sheet.cell_value(i+1,8))#right knee z coordinate
            ccp[0] = float(sheet.cell_value(i+1,9))#right foot x coordinate
            ccp[1] = float(sheet.cell_value(i+1,10))#right foot y coordinate
            ccp[2] = float(sheet.cell_value(i+1,11))#right foot z coordinate
            angel_tem=cal_angle(aap, bbp, ccp)
            sheet2.write(i+1,2,angel_tem)#Calculate angle (degrees)

            #The left knee angle is calculated here (L_KNEE_ANGEL)
            aap[0] = float(sheet.cell_value(i+1,12))#left hip x coordinate
            aap[1] = float(sheet.cell_value(i+1,13))#left hip y coordinate
            aap[2] = float(sheet.cell_value(i+1,14))#left hip z coordinate
            bbp[0] = float(sheet.cell_value(i+1,15))#left knee x coordinate
            bbp[1] = float(sheet.cell_value(i+1,16))#left knee y coordinate
            bbp[2] = float(sheet.cell_value(i+1,17))#left knee z coordinate
            ccp[0] = float(sheet.cell_value(i+1,18))#left foot x coordinate
            ccp[1] = float(sheet.cell_value(i+1,19))#left foot y coordinate
            ccp[2] = float(sheet.cell_value(i+1,20))#left foot z coordinate
            angel_tem=cal_angle(aap, bbp, ccp)
            sheet2.write(i+1,4,angel_tem)#Calculate angle (degrees)

            #Calculate the spine curvature angle here (SPINE_ANGEL)
            aap[0] = float(sheet.cell_value(i+1,24))#Chest x coordinate
            aap[1] = float(sheet.cell_value(i+1,25))#Chest y coordinate
            aap[2] = float(sheet.cell_value(i+1,26))#Chest z coordinate
            bbp[0] = float(sheet.cell_value(i+1,21))#spine x coordinate
            bbp[1] = float(sheet.cell_value(i+1,22))#spine y coordinate
            bbp[2] = float(sheet.cell_value(i+1,23))#spine z coordinate
            ccp[0] = float(sheet.cell_value(i+1,0))#center x coordinate
            ccp[1] = float(sheet.cell_value(i+1,1))#center y coordinate
            ccp[2] = float(sheet.cell_value(i+1,2))#center z coordinate
            angel_tem=cal_angle(aap, bbp, ccp)
            sheet2.write(i+1,5,angel_tem)#Calculate angle (degrees)




    savepath = path + angel_num + '_angel.xls'
    book.save(savepath)


#angel_all('1','./outputs/')
