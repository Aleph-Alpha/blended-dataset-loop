use std::f64;
use argminmax::ArgMinMax;  // import trait
use std::fs::File;
use std::io::{BufWriter, Write};
use tqdm::tqdm;
use serde::Serialize;

#[derive(Serialize)]
struct Result {
    number_to_sample: Vec<usize>,
    result: Vec<Vec<usize>>
}

fn sample(number_to_sample: Vec<usize>, filename: &str) {

    let mut number_sampled: Vec<usize> = vec![0; number_to_sample.len()];
    let mut proportion_sampled: Vec<f64> = vec![0.0; number_to_sample.len()];

    let total_count: usize = number_to_sample.iter().copied().sum::<usize>();

    let mut result: Vec<Vec<usize>> = Vec::with_capacity(total_count);

    for _ in tqdm(0..total_count) {
        
        // find smallest represenation
        let (dataset_index, _argmax) = proportion_sampled.argminmax();  


        // add representation from dataset
        let index_next =  number_sampled[dataset_index];

        // update proportions
        number_sampled[dataset_index] += 1;
        proportion_sampled[dataset_index] = (number_sampled[dataset_index] as f64) / number_to_sample[dataset_index] as f64;

        result.push(vec![dataset_index, index_next]);

    }

    assert!(total_count == number_sampled.iter().copied().sum::<usize>());
    number_to_sample.iter().zip(number_sampled.iter()).for_each(|(a, b)| assert_eq!(a, b));


    // write
    let file = File::create(filename).unwrap();
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &Result {
        number_to_sample, result
    }).unwrap();
    writer.flush().unwrap();
}

fn main() {

    let number_to_sample_2048: Vec<usize> = vec![12134005, 19527638, 274, 845109, 453, 527, 305814, 7600361, 232376, 299349, 88637, 211976, 11997007, 10000470, 76195, 222639, 55737, 39365, 1567916, 26869, 175635, 10604, 30135, 62, 1584, 2384, 1119, 6972, 26731, 27, 4750, 944, 11516, 3783, 58640, 455, 495819, 8120, 301, 13755, 207188, 4664904, 23217681, 2228511, 34794, 30130, 35968, 38632, 34473, 36185, 42648, 7149519, 148535, 153327, 61876, 1810938, 32205, 159192, 49844, 51456, 14249, 361617, 1226437, 1584293, 110664, 1141209, 47, 1596258, 21536, 374165, 1780050, 59705, 126734, 178832, 217456, 15397, 7302742, 502, 28965, 136, 153, 796, 850815, 32132, 900093, 1296187, 147530, 769393, 506665, 175, 31436417, 385, 2248, 32, 231, 6010, 194940, 7320, 10697, 17693, 31512, 2257981, 25831, 7990, 17188, 121, 1008932, 1618, 88, 11, 47, 29, 3218, 14784131, 4327, 31583, 1041, 2075, 2252, 59989, 13033725, 191100, 266751, 64371, 6884, 14641, 52022, 33495035, 196048, 70545, 18747, 21966, 63203, 4399, 11246084, 92517, 3851, 1872, 1195, 394386, 72090, 3621, 1579, 5282855, 1601, 523, 23476, 44076, 559993, 463115, 298935, 191662, 126963, 347498, 369283, 135338, 95229, 1108692, 99630, 603693, 924280, 423387, 54358, 35135, 1111216, 726047, 27925, 371814, 1208, 29537, 176863, 353604, 72474, 553626, 407845, 863788, 153032, 187605, 2776215, 7112, 38765, 5318, 7444, 10272, 2541, 4612, 1619, 4055, 2000, 743, 893, 654, 1233233, 2874273, 743560, 1083350, 671388, 312647, 498100, 7801, 14427, 3173, 2185, 4016, 818, 780, 91042, 95610, 92512, 120570, 56320, 95737, 39551, 18572, 29679, 14473, 21221, 21466, 73651, 77052, 50126, 76208, 65515, 129122, 42927, 87239, 215614, 28389, 87052, 51420, 82082, 73094, 128406, 1325356, 1847423, 1591204, 2577291, 2286177, 2879338, 2572738, 2370572, 2713983, 2736689, 2682314, 2859103, 2397835, 2572453, 2923437, 2363180, 2725627, 2792097, 2596249, 3321366, 3064605, 3149566, 2794492, 2885005, 2966593, 3005319, 3042711, 3309377, 2915736, 3411285, 2979901, 2886321, 3863920, 2838460, 3735454, 3110212, 3905125, 2930295, 3919212, 3044050, 3027869, 3922025, 3023370, 3746744, 2961879, 2657320, 3884765, 3657518, 4429013, 3015725, 3813323, 4457491, 4070278, 3068493, 4507105, 4750901, 5070736, 4845473, 6328115, 7283866, 8317295, 4145558, 5982183, 2677390, 1349305, 2013662, 1076974, 1432698, 1523792, 4865123, 4296510, 2910216, 4790914, 3465711, 3762940, 1166091, 2756646, 5965010, 728892, 1342000, 769103, 1059665, 1177535, 1915768, 18633596, 26835353, 22400348, 31711915, 25177558, 32902252, 27018070, 28544767, 26066504, 28742486, 25692094, 24631437, 22079006, 24863980, 27450154, 17397743, 19611332, 20323612, 18597526, 23070370, 21650844, 21906396, 18719323, 19396665, 20532435, 20751918, 21141152, 23294817, 20890614, 22349830, 19687009, 18502590, 25044510, 17112244, 23204684, 19177880, 25181100, 18450914, 25502250, 19964510, 19590883, 24959818, 19811124, 24713784, 19664096, 18174813, 26652276, 24965358, 29647036, 19117392, 24470669, 30133835, 28106721, 20223028, 28393084, 30364042, 31834551, 30735225, 156567, 148187, 196960, 86451, 160638, 65922, 27829, 48022, 23930, 31884, 36404, 109114, 98582, 77290, 114371, 90919, 108666, 24817, 69010, 194841, 26258, 70484, 51384, 80433, 88728, 244636, 2305902, 3386118, 2826796, 5180244, 3874926, 5168585, 4218593, 4472958, 3927470, 4568774, 4130896, 3606704, 3550799, 4005317, 5014142, 2540338, 3115925, 2994946, 2863050, 3270410, 3413621, 3244517, 2798960, 2803022, 3041132, 3108305, 3141127, 3417463, 2991804, 3342706, 2913431, 2821663, 3801869, 2750849, 3520391, 3190051, 4024943, 2921000, 4108364, 3040832, 2978249, 3948221, 3074521, 3824810, 3058990, 2832461, 4009371, 3793731, 4370392, 2935201, 3703045, 4411699, 4182720, 3303852, 4529004, 4929066, 5246502, 5027778, 96430, 91918, 122548, 53681, 104707, 41261, 16612, 27647, 13611, 18748, 18853, 64756, 59748, 40549, 67157, 53871, 79910, 24512, 50731, 133372, 18918, 61792, 48230, 73826, 75361, 153306, 1520979, 2225365, 1857659, 3280802, 2798210, 3484960, 2896069, 2774422, 2807696, 2990233, 2765282, 2765241, 2472988, 2792802, 3197922, 2376355, 2700676, 2944233, 2639638, 3214707, 3063444, 2948937, 2587932, 2615308, 2785338, 2807828, 2813385, 3086397, 2793429, 3109241, 2739412, 2672726, 3620782, 2663468, 3473029, 2912226, 3701629, 2792169, 3753465, 2908782, 2864040, 3828451, 2975711, 3636301, 2812099, 2605013, 3850185, 3624632, 4178259, 2832678, 3602138, 4098580, 3943563, 2974272, 4380739, 4742262, 4911677, 4720854, 41416, 38374, 48733, 23285, 42102, 15255, 6265, 10775, 6650, 7428, 8567, 23988, 22566, 15826, 24692, 20528, 30783, 9976, 19159, 52671, 9966, 38923, 23748, 41867, 37629, 97649, 934872, 1367151, 1164913, 1866026, 1533061, 1953756, 1658351, 1676916, 1599981, 1752886, 1677376, 1493489, 1389480, 1552952, 1908786, 1187756, 1365944, 1458194, 1316492, 1643947, 1668622, 1689951, 1496794, 1509015, 1682305, 1716751, 1748742, 1874659, 1657072, 1798192, 1625599, 1548592, 2122463, 1524554, 2050484, 1783593, 2303102, 1643843, 2344462, 1720444, 1710634, 2279025, 1742863, 2182077, 1742182, 1599465, 2310510, 2155080, 2561789, 1699731, 2167006, 2545718, 2409316, 1876110, 2633621, 2866318, 3002963, 2932455];
    let number_to_sample_4096: Vec<usize> = vec![6067004, 9763821, 136, 422554, 225, 262, 152907, 3800180, 116188, 149674, 44318, 105988, 5998502, 5000235, 38097, 111320, 27868, 19682, 783958, 13435, 87818, 5302, 15067, 31, 792, 1192, 558, 3486, 13365, 13, 2375, 472, 5758, 1891, 29320, 227, 247909, 4060, 150, 6877, 103594, 2332451, 11608841, 1114255, 17397, 15064, 17984, 19316, 17236, 18092, 21324, 3574759, 74267, 76663, 30938, 905469, 16103, 79596, 24922, 25728, 7125, 180808, 613219, 792146, 55332, 570604, 23, 798129, 10768, 187082, 890025, 29852, 63367, 89416, 108728, 7699, 3651371, 251, 14483, 68, 76, 397, 425407, 16066, 450046, 648093, 73765, 384696, 253332, 87, 15718210, 192, 1124, 16, 115, 3004, 97470, 3659, 5348, 8846, 15755, 1128991, 12914, 3995, 8594, 60, 504466, 809, 44, 5, 23, 14, 1609, 7392064, 2163, 15791, 519, 1037, 1126, 29994, 6516862, 95550, 133376, 32185, 3442, 7319, 26011, 16747519, 98024, 35272, 9373, 10983, 31600, 2199, 5623043, 46258, 1924, 936, 597, 197193, 36045, 1809, 789, 2641428, 800, 260, 11738, 22038, 279996, 231557, 149467, 95830, 63481, 173749, 184641, 67668, 47614, 554346, 49814, 301846, 462139, 211693, 27179, 17567, 555608, 363023, 13961, 185906, 604, 14768, 88431, 176802, 36236, 276813, 203922, 431893, 76516, 93802, 1388108, 3556, 19382, 2659, 3722, 5135, 1270, 2306, 809, 2028, 1000, 372, 446, 327, 616616, 1437137, 371780, 541675, 335694, 156324, 249050, 3900, 7214, 1586, 1092, 2008, 409, 390, 45521, 47805, 46256, 60285, 28160, 47868, 19776, 9286, 14839, 7236, 10610, 10733, 36825, 38526, 25063, 38104, 32757, 64561, 21464, 43619, 107807, 14194, 43526, 25710, 41041, 36546, 64203, 662678, 923711, 795602, 1288645, 1143087, 1439668, 1286369, 1185286, 1356991, 1368343, 1341157, 1429551, 1198918, 1286227, 1461718, 1181590, 1362814, 1396048, 1298125, 1660683, 1532302, 1574783, 1397246, 1442503, 1483296, 1502660, 1521356, 1654689, 1457868, 1705642, 1489951, 1443161, 1931960, 1419230, 1867727, 1555106, 1952562, 1465148, 1959606, 1522025, 1513935, 1961013, 1511685, 1873372, 1480939, 1328659, 1942383, 1828759, 2214507, 1507862, 1906662, 2228746, 2035139, 1534246, 2253552, 2375450, 2535368, 2422737, 3164058, 3641932, 4158647, 2072779, 2991091, 1338695, 674652, 1006831, 538487, 716349, 761896, 2432561, 2148255, 1455109, 2395457, 1732856, 1881470, 583045, 1378323, 2982505, 364446, 671000, 384551, 529833, 588767, 957884, 9316798, 13417677, 11200175, 15855959, 12588781, 16451128, 13509036, 14272385, 13033254, 14371245, 12846048, 12315720, 11039504, 12431991, 13725079, 8698873, 9805667, 10161807, 9298764, 11535186, 10825423, 10953198, 9359662, 9698334, 10266219, 10375961, 10570577, 11647410, 10445308, 11174916, 9843506, 9251295, 12522255, 8556122, 11602342, 9588941, 12590552, 9225457, 12751127, 9982256, 9795443, 12479911, 9905563, 12356894, 9832049, 9087408, 13326140, 12482681, 14823519, 9558697, 12235336, 15066919, 14053362, 10111516, 14196544, 15182023, 15917278, 15367615, 78284, 74093, 98479, 43224, 80319, 32961, 13915, 24011, 11965, 15942, 18202, 54557, 49291, 38645, 57185, 45459, 54333, 12408, 34505, 97419, 13129, 35241, 25692, 40216, 44364, 122318, 1152951, 1693058, 1413398, 2590122, 1937463, 2584292, 2109296, 2236479, 1963735, 2284388, 2065448, 1803352, 1775400, 2002657, 2507071, 1270169, 1557963, 1497473, 1431525, 1635205, 1706811, 1622258, 1399480, 1401511, 1520567, 1554152, 1570563, 1708732, 1495901, 1671353, 1456716, 1410831, 1900934, 1375425, 1760196, 1595025, 2012471, 1460500, 2054183, 1520416, 1489125, 1974111, 1537260, 1912405, 1529495, 1416231, 2004685, 1896865, 2185196, 1467601, 1851523, 2205848, 2091360, 1651926, 2264502, 2464533, 2623251, 2513889, 48214, 45958, 61273, 26840, 52354, 20629, 8306, 13823, 6805, 9374, 9426, 32377, 29874, 20274, 33577, 26935, 39955, 12256, 25366, 66686, 9459, 30896, 24114, 36913, 37680, 76653, 760489, 1112683, 928829, 1640401, 1399105, 1742480, 1448034, 1387211, 1403848, 1495117, 1382640, 1382620, 1236494, 1396401, 1598961, 1188178, 1350338, 1472117, 1319819, 1607353, 1531722, 1474469, 1293966, 1307654, 1392668, 1403913, 1406692, 1543198, 1396714, 1554621, 1369706, 1336363, 1810391, 1331734, 1736514, 1456113, 1850815, 1396085, 1876732, 1454391, 1432020, 1914226, 1487856, 1818151, 1406050, 1302506, 1925093, 1812316, 2089129, 1416339, 1801068, 2049290, 1971782, 1487136, 2190369, 2371131, 2455839, 2360427, 20708, 19187, 24365, 11642, 21051, 7627, 3132, 5387, 3324, 3714, 4283, 11994, 11282, 7912, 12346, 10264, 15392, 4988, 9579, 26335, 4983, 19461, 11874, 20933, 18814, 48824, 467436, 683575, 582457, 933013, 766530, 976877, 829176, 838457, 799991, 876443, 838688, 746743, 694740, 776476, 954393, 593878, 682972, 729097, 658246, 821972, 834311, 844974, 748397, 754507, 841153, 858375, 874371, 937330, 828536, 899096, 812799, 774296, 1061231, 762276, 1025242, 891797, 1151551, 821921, 1172231, 860222, 855317, 1139513, 871432, 1091037, 871091, 799733, 1155255, 1077539, 1280895, 849865, 1083503, 1272859, 1204658, 938055, 1316810, 1433159, 1501482, 1466227];
    let number_to_sample_8192: Vec<usize> = vec![3033502, 4881911, 68, 211276, 113, 131, 76453, 1900090, 58094, 74837, 22159, 52993, 2999251, 2500117, 19048, 55659, 13934, 9841, 391979, 6717, 43909, 2651, 7532, 15, 396, 596, 279, 1743, 6682, 5, 1187, 235, 2879, 944, 14660, 113, 123954, 2030, 75, 3438, 51797, 1166226, 5804420, 557127, 8697, 7532, 8992, 9658, 8617, 9046, 10662, 1787380, 37133, 38331, 15469, 452733, 8051, 39798, 12461, 12864, 3562, 90403, 306609, 396073, 27666, 285302, 11, 399063, 5384, 93540, 445012, 14925, 31683, 44708, 54364, 3848, 1825685, 125, 7241, 34, 38, 199, 212703, 8032, 225023, 324046, 36882, 192347, 126666, 43, 7859106, 95, 562, 8, 57, 1502, 48735, 1830, 2674, 4423, 7878, 564496, 6457, 1997, 4297, 30, 252232, 404, 22, 2, 11, 7, 804, 3696033, 1081, 7894, 260, 517, 563, 14996, 3258431, 47775, 66687, 16091, 1721, 3660, 13005, 8373759, 49012, 17635, 4686, 5491, 15800, 1098, 2811521, 23129, 962, 468, 298, 98595, 18022, 904, 394, 1320714, 400, 130, 5869, 11018, 139998, 115778, 74733, 47914, 31739, 86874, 92320, 33834, 23807, 277173, 24906, 150923, 231070, 105846, 13589, 8783, 277804, 181511, 6981, 92953, 302, 7384, 44214, 88401, 18117, 138406, 101961, 215946, 38258, 46901, 694054, 1778, 9691, 1329, 1861, 2568, 635, 1153, 404, 1014, 500, 186, 223, 163, 308308, 718569, 185890, 270837, 167846, 78162, 124525, 1949, 3606, 793, 546, 1004, 204, 195, 22760, 23902, 23128, 30142, 14080, 23934, 9888, 4643, 7418, 3617, 5305, 5366, 18412, 19263, 12531, 19052, 16378, 32280, 10732, 21809, 53903, 7097, 21763, 12854, 20520, 18273, 32101, 331338, 461856, 397801, 644323, 571544, 719833, 643184, 592643, 678496, 684172, 670578, 714775, 599459, 643113, 730859, 590795, 681407, 698024, 649062, 830341, 766150, 787391, 698622, 721251, 741648, 751330, 760677, 827343, 728934, 852821, 744975, 721580, 965980, 709615, 933863, 777553, 976281, 732574, 979803, 761011, 756966, 980506, 755842, 936686, 740469, 664330, 971191, 914379, 1107253, 753930, 953331, 1114372, 1017569, 767123, 1126776, 1187725, 1267684, 1211368, 1582029, 1820967, 2079324, 1036388, 1495546, 669347, 337326, 503416, 269243, 358174, 380948, 1216281, 1074127, 727554, 1197728, 866428, 940735, 291523, 689161, 1491253, 182223, 335500, 192275, 264916, 294384, 478942, 4658400, 6708839, 5600088, 7927980, 6294391, 8225563, 6754518, 7136193, 6516626, 7185623, 6423023, 6157860, 5519752, 6215996, 6862540, 4349437, 4902834, 5080904, 4649382, 5767593, 5412712, 5476600, 4679830, 4849167, 5133109, 5187981, 5285289, 5823704, 5222653, 5587458, 4921753, 4625648, 6261127, 4278062, 5801172, 4794471, 6295276, 4612729, 6375563, 4991128, 4897721, 6239955, 4952782, 6178447, 4916025, 4543704, 6663070, 6241341, 7411760, 4779349, 6117667, 7533460, 7026681, 5055758, 7098272, 7591012, 7958640, 7683808, 39142, 37046, 49239, 21612, 40159, 16480, 6957, 12005, 5982, 7971, 9101, 27277, 24645, 19322, 28592, 22729, 27166, 6204, 17252, 48710, 6564, 17620, 12845, 20108, 22182, 61159, 576475, 846529, 706699, 1295061, 968731, 1292146, 1054647, 1118239, 981867, 1142194, 1032724, 901676, 887700, 1001329, 1253535, 635084, 778981, 748736, 715762, 817602, 853405, 811129, 699740, 700755, 760283, 777076, 785281, 854366, 747951, 835675, 728358, 705415, 950467, 687712, 880098, 797513, 1006236, 730250, 1027091, 760208, 744562, 987055, 768630, 956202, 764747, 708115, 1002343, 948432, 1092597, 733800, 925761, 1102925, 1045679, 825963, 1132251, 1232267, 1311625, 1256945, 24107, 22978, 30637, 13420, 26177, 10315, 4152, 6911, 3402, 4687, 4713, 16189, 14937, 10137, 16789, 13467, 19977, 6128, 12683, 33343, 4729, 15448, 12057, 18456, 18840, 38326, 380245, 556340, 464413, 820199, 699552, 871240, 724017, 693606, 701924, 747558, 691320, 691310, 618247, 698200, 799480, 594089, 675169, 736058, 659909, 803676, 765861, 737233, 646983, 653827, 696334, 701957, 703346, 771599, 698357, 777310, 684853, 668181, 905195, 665867, 868257, 728056, 925406, 698042, 938365, 727195, 716010, 957113, 743927, 909075, 703025, 651253, 962545, 906158, 1044565, 708169, 900535, 1024645, 985891, 743568, 1095185, 1185565, 1227919, 1180213, 10354, 9593, 12183, 5820, 10525, 3812, 1566, 2693, 1662, 1856, 2141, 5997, 5641, 3956, 6172, 5132, 7696, 2494, 4789, 13167, 2491, 9730, 5937, 10466, 9407, 24411, 233718, 341787, 291228, 466506, 383265, 488439, 414588, 419229, 399995, 438221, 419344, 373372, 347370, 388238, 477196, 296939, 341486, 364548, 329123, 410987, 417155, 422487, 374198, 377252, 420576, 429188, 437185, 468665, 414268, 449548, 406399, 387148, 530615, 381138, 512620, 445898, 575775, 410961, 586115, 430111, 427658, 569756, 435716, 545518, 435545, 399866, 577627, 538770, 640447, 424932, 541751, 636429, 602329, 469027, 658405, 716579, 750741, 733114];
    assert_eq!(number_to_sample_2048.len(), number_to_sample_4096.len());
    assert_eq!(number_to_sample_2048.len(), number_to_sample_8192.len());
    number_to_sample_2048.iter().zip(number_to_sample_4096.iter()).for_each(|(a, b)| assert!(a > b));
    number_to_sample_4096.iter().zip(number_to_sample_8192.iter()).for_each(|(a, b)| assert!(a > b));

    let example: Vec<usize> = vec![1, 2, 3, 4];
    
    sample(example, "/pfss/alephalpha/dataset-v2/tokenized_unigram_128k_multilingual/blended_dataset_tmp_cache/example.json");
    sample(number_to_sample_2048, "/pfss/alephalpha/dataset-v2/tokenized_unigram_128k_multilingual/blended_dataset_tmp_cache/2048.json");
    sample(number_to_sample_4096, "/pfss/alephalpha/dataset-v2/tokenized_unigram_128k_multilingual/blended_dataset_tmp_cache/4096.json");
    sample(number_to_sample_8192, "/pfss/alephalpha/dataset-v2/tokenized_unigram_128k_multilingual/blended_dataset_tmp_cache/8192.json");
   
}