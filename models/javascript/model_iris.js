class Model{
    constructor(){
        // (1) code_layers_class_instance
        this.layers = [ new Dense(softsign, [ new Matrix([[0.04962513968348503, 0.12553013861179352, 0.3046343922615051, -0.16916309297084808, 0.05414486303925514, 0.2674271762371063, 0.2647090554237366, -0.1515604704618454, -0.07715032249689102, 0.32342496514320374, 0.11532612890005112, -0.06974746286869049, 0.009173242375254631, 0.13914649188518524, 0.05223739147186279, -0.12089597433805466, 0.030794156715273857, 0.09342066943645477, -0.22587546706199646, 0.11709319800138474, -0.047322243452072144, 0.11136350780725479, 0.13000819087028503, -0.0925510823726654, -0.16738630831241608, -0.1378757655620575, -0.07219097018241882, 0.3506162762641907, 0.08762899041175842, -0.008010773919522762, 0.2749738097190857, -0.09860825538635254, -0.0252248402684927, -0.3898336887359619, -0.09243220090866089, -0.2858728766441345, -0.05273061990737915, 0.25102218985557556, -0.17686167359352112, -0.1372833102941513, 0.23590610921382904, 0.03908121958374977, -0.2384660840034485, -0.06538426876068115, 0.13307419419288635, -0.07786110788583755, -0.08148251473903656, -0.028347589075565338, 0.08453478664159775, 0.26971009373664856, 0.15808118879795074, -0.21979212760925293, -0.14671191573143005, 0.3890821933746338, -0.15230832993984222, -0.06826201826334, -0.05677938833832741, -0.09089088439941406, 0.1015559732913971, -0.05839477479457855, -0.14125986397266388, 0.19908177852630615, 0.02870159223675728, 0.017928894609212875, 0.2582249641418457, 0.25567224621772766, -0.06815556436777115, 0.10232434421777725, -0.32450324296951294, -0.06416217237710953, -0.13468505442142487, -0.449273020029068, -0.19371463358402252, 0.23478782176971436, -0.11294372379779816, -0.19190442562103271, 0.2551276981830597, -0.2535836696624756, -0.3523270785808563, -0.048650141805410385, -0.12283653765916824, -0.003061911091208458, 0.3310486972332001, -0.22076857089996338, -0.05338058993220329, -0.11718300729990005, 0.12179315090179443, 0.14426346123218536, -0.16955214738845825, 0.22835464775562286, 0.0893324464559555, 0.01450114231556654, 0.10873168706893921, 0.08597782254219055, 0.16063973307609558, -0.10757800191640854, -0.01897694543004036, -0.02096296101808548, -0.0646408274769783, -0.3208092153072357], [0.0009011677466332912, -0.4350764751434326, -0.6323554515838623, 0.07065826654434204, -0.04515044018626213, 0.08307566493749619, -0.4378353953361511, 0.5905300378799438, 0.22011172771453857, -0.02994540147483349, -0.06672851741313934, -0.07746965438127518, -0.0612596720457077, -0.3939482867717743, -0.08779214322566986, -0.5391917824745178, -0.42001399397850037, -0.6215128898620605, 0.45190900564193726, -0.5025970339775085, 0.16140124201774597, 0.002567857736721635, -0.44122612476348877, -0.008158266544342041, 0.4545195996761322, 0.5023253560066223, 0.11304442584514618, -0.6468881368637085, -0.1079094409942627, -0.4054913818836212, -0.4517071545124054, -0.3249850273132324, -0.2674557566642761, 0.6135304570198059, -0.01952601969242096, 0.38303714990615845, 0.33844271302223206, -0.7242711782455444, 0.19766750931739807, 0.47223931550979614, -0.39584049582481384, 0.07432013750076294, 0.24205975234508514, 0.24598689377307892, 0.338355153799057, -0.07287800312042236, 0.47019705176353455, 0.4584588408470154, -0.32416290044784546, 0.11345549672842026, -0.024437634274363518, -0.16366928815841675, 0.4833916425704956, -0.35011589527130127, 0.5129174590110779, 0.29617998003959656, -0.12854406237602234, 0.4259157180786133, -0.10565309226512909, 0.44944605231285095, 0.12814298272132874, -0.6371158361434937, 0.011260390281677246, 0.49616938829421997, 0.26580896973609924, -0.30919262766838074, 0.4176730513572693, 0.1455487310886383, 0.09519397467374802, 0.3067394495010376, 0.12176480889320374, 0.3419681787490845, -0.22722263634204865, -0.5256900787353516, 0.24304291605949402, 0.7742596864700317, 0.08339861780405045, 0.23975729942321777, 0.6415854096412659, 0.25593218207359314, 0.2833654582500458, 0.17505034804344177, -0.6827571392059326, 0.539825975894928, 0.09773654490709305, 0.0052446406334638596, -0.7355985045433044, -0.08410397917032242, -0.21650883555412292, -0.45174193382263184, 0.1321839690208435, -0.22669988870620728, 0.08528074622154236, 0.1728905737400055, -0.13020825386047363, 0.2535056471824646, -0.21576809883117676, -0.543859601020813, 0.4953983724117279, 0.14578503370285034], [-0.0641397014260292, 0.34666672348976135, 0.2929830849170685, -0.036686740815639496, -0.1676214337348938, -0.1511273980140686, 0.40984347462654114, -0.4132227897644043, -0.5047429203987122, 0.05946842581033707, -0.17411495745182037, 0.25089406967163086, 0.009587962180376053, 0.36632293462753296, 0.3690459728240967, 0.30657410621643066, 0.14537164568901062, 0.5663006901741028, -0.2407034933567047, 0.6380149126052856, -0.19479548931121826, -0.3296300172805786, 0.353923499584198, -0.24111899733543396, -0.4530416429042816, -0.28969016671180725, -0.06271840631961823, 0.16026967763900757, 0.29662269353866577, 0.3490280210971832, 0.5137108564376831, -0.12505683302879333, 0.31440919637680054, -0.22685974836349487, 0.32885032892227173, -0.6582220792770386, -0.2678307294845581, 0.43377572298049927, -0.2126278579235077, -0.5472832918167114, 0.46031269431114197, -0.4525960087776184, -0.08206325769424438, -0.23590901494026184, -0.24877269566059113, 0.12221881002187729, -0.262107253074646, -0.4034152328968048, 0.32409366965293884, -0.19850818812847137, -0.2713788151741028, 0.22937525808811188, -0.20071782171726227, 0.25448694825172424, -0.30438318848609924, 0.04389822855591774, -0.056954555213451385, -0.31592684984207153, -0.15030483901500702, -0.34437984228134155, -0.06868050247430801, 0.6558053493499756, 0.19746430218219757, -0.546755313873291, 0.13478830456733704, 0.23338152468204498, -0.41114547848701477, -0.25767847895622253, -0.18435163795948029, -0.24963343143463135, -0.4195546507835388, -0.20508189499378204, -0.1480850726366043, 0.6895024180412292, -0.051816072314977646, -0.2890721559524536, 0.04382136091589928, -0.07385193556547165, -0.2031848281621933, -0.18178033828735352, -0.16037613153457642, -0.02074126899242401, 0.29334551095962524, -0.4033139944076538, -0.2722773253917694, -0.24686592817306519, 0.19401243329048157, -0.08487824350595474, 0.4574767053127289, 0.25505778193473816, -0.0488320030272007, 0.17105703055858612, 0.16398827731609344, -0.12063644081354141, 0.23995286226272583, -0.3254563808441162, 0.27322372794151306, 0.4824274778366089, -0.26292896270751953, -0.31726497411727905], [-0.20970883965492249, 0.5648207068443298, 0.6159176230430603, 0.196104034781456, -0.25815653800964355, 0.1477247178554535, 0.5831465721130371, -0.33677586913108826, -0.36284637451171875, 0.1551710069179535, -0.4217122793197632, 0.28783026337623596, -0.1791064292192459, 0.6365050673484802, 0.489577978849411, 0.5865697264671326, 0.519062876701355, 0.650537371635437, -0.3499942421913147, 0.42680323123931885, -0.19331400096416473, -0.3731016516685486, 0.5585256218910217, -0.14362117648124695, -0.28309231996536255, -0.6464081406593323, -0.060838814824819565, 0.6890891194343567, 0.4612080156803131, 0.7500118613243103, 0.27479782700538635, 0.0060342345386743546, 0.39229264855384827, -0.47151559591293335, 0.5220029354095459, -0.29601821303367615, -0.42011305689811707, 0.7760151624679565, -0.11265948414802551, -0.6212873458862305, 0.37618786096572876, -0.5156416893005371, -0.5134949684143066, -0.40040504932403564, -0.5158432126045227, 0.11721526086330414, -0.6950461864471436, -0.1416640728712082, 0.4850137531757355, 0.018484776839613914, -0.054083775728940964, 0.17378339171409607, -0.5279465317726135, 0.1598319709300995, -0.48602214455604553, -0.16663290560245514, -0.01031053252518177, -0.6275069713592529, -0.05067960172891617, -0.700812578201294, -0.44393420219421387, 0.7205184102058411, 0.04808599874377251, -0.374881386756897, -0.13882336020469666, 0.37943774461746216, -0.5553862452507019, -0.15639276802539825, 0.17737142741680145, -0.7254559993743896, -0.27068841457366943, -0.21282239258289337, -0.14082938432693481, 0.21866919100284576, -0.35869279503822327, -0.7115353941917419, 0.34944280982017517, -0.6162984371185303, -0.7182255387306213, 0.2913096249103546, -0.24608078598976135, -0.10836204141378403, 0.3242514431476593, -0.48487308621406555, -0.5393898487091064, -0.21457204222679138, 0.6582335829734802, -0.4343746602535248, 0.32195186614990234, 0.12733744084835052, 0.33477485179901123, 0.08755097538232803, 0.2758368253707886, -0.400112509727478, 0.4010216295719147, -0.3683196008205414, 0.20280110836029053, 0.7611461877822876, -0.3576086163520813, 0.025776492431759834]]), new Matrix([[-0.13916294276714325, -0.487104207277298, -0.4955674707889557, -0.02022349275648594, 0.285134494304657, 0.1870768815279007, -0.639252781867981, -0.3470042645931244, 0.5345346331596375, -0.18497762084007263, 0.2667466700077057, -0.035367660224437714, 0.08253303170204163, -0.6079486012458801, -0.5532523989677429, -0.18061508238315582, -0.28136202692985535, -0.6290124654769897, 0.09782035648822784, -0.15795119106769562, 0.30471402406692505, 0.4882256090641022, -0.36773964762687683, -0.44147953391075134, -0.06607180833816528, 0.49341467022895813, 0.0005556595278903842, -0.46989336609840393, 0.29116204380989075, -0.5625225901603699, -0.4043736457824707, -0.6913496255874634, -0.25469428300857544, 0.3459441363811493, -0.4695208966732025, 0.08293359726667404, 0.053585875779390335, -0.3985443115234375, 0.03720241039991379, 0.6719260215759277, -0.46765342354774475, 0.5389952063560486, 0.5079752206802368, 0.39389705657958984, 0.21746721863746643, -0.2701718807220459, 0.4996567368507385, -0.05467841029167175, -0.10871824622154236, -0.09362601488828659, 0.14521701633930206, -0.2662747800350189, 0.14402243494987488, -0.16263693571090698, 0.2664410173892975, 0.39014264941215515, 0.014778541401028633, 0.49746009707450867, -0.25692611932754517, 0.56051105260849, 0.3785240054130554, -0.19719558954238892, 0.26564815640449524, -0.08326394110918045, 0.29714763164520264, -0.3336201310157776, 0.5094192028045654, 0.19961665570735931, -0.19248703122138977, 0.5813129544258118, -0.48747968673706055, -0.033675529062747955, 0.4247421324253082, 0.29741370677948, 0.14779986441135406, 0.42791691422462463, -0.4614802300930023, 0.5211275219917297, 0.5549717545509338, -0.22796590626239777, 0.07381194084882736, -0.2689710557460785, -0.1476239264011383, 0.4636254608631134, 0.5347223281860352, 0.36069992184638977, -0.2062019407749176, 0.34613358974456787, -0.31531432271003723, -0.036731988191604614, -0.3960478603839874, 0.0866047590970993, -0.3372906446456909, 0.39215365052223206, -0.40651100873947144, 0.43389496207237244, -0.28619953989982605, -0.6234196424484253, 0.030658511444926262, -0.36899223923683167]]) ]), new Dense(softmax, [ new Matrix([[0.05422789603471756, 0.01935596577823162, 0.04172999784350395], [-0.968827486038208, -0.2029765099287033, 0.48171916604042053], [-0.6804401278495789, -0.12478172034025192, 0.6109287738800049], [-0.13723966479301453, -0.024895407259464264, -0.002379385055974126], [-0.3326002359390259, 0.3491750657558441, -0.05604971945285797], [-0.2891019284725189, 0.2725031077861786, 0.20523783564567566], [-0.8306269645690918, -0.32703498005867004, 0.682074248790741], [0.9565001726150513, -0.25017303228378296, -0.11117318272590637], [0.14114165306091309, 0.3298337459564209, -0.4423486888408661], [-0.4045468866825104, -0.07939717918634415, 0.05912657082080841], [0.6332624554634094, 0.21778932213783264, -0.19706250727176666], [-0.6268684267997742, 0.07436737418174744, 0.26876410841941833], [-0.025472678244113922, -0.08561651408672333, -0.13792504370212555], [-0.6520407199859619, -0.5552405714988708, 0.6876947283744812], [-0.09702339768409729, -0.1803942322731018, 0.5170612335205078], [-0.8523861765861511, -0.02278200536966324, 0.3256680369377136], [-0.20583514869213104, 0.035400841385126114, 0.3498590886592865], [-0.6085935831069946, -0.0715426355600357, 0.7616155743598938], [0.5033997893333435, -0.3275984823703766, -0.4978446662425995], [-1.0713019371032715, 0.17388710379600525, 0.36797070503234863], [-0.3483196496963501, 0.13611117005348206, -0.09364823997020721], [-0.07900796085596085, 0.35787203907966614, -0.22622114419937134], [-1.1615842580795288, 0.02707667276263237, 0.538853645324707], [0.2960713803768158, -0.4946812093257904, -0.04522128403186798], [0.6790716052055359, -0.06505478918552399, -0.16910937428474426], [0.8540053367614746, 0.1281355917453766, -0.5455893874168396], [0.2326056957244873, -0.0014412249438464642, 0.004953172989189625], [-0.5332292914390564, 0.011577317491173744, 0.6887134313583374], [-0.7564554810523987, 0.26439762115478516, 0.11586602032184601], [-0.8695017099380493, -0.3294334411621094, 0.6969726085662842], [-0.7445091009140015, -0.13409645855426788, 0.5482024550437927], [0.3009323179721832, -0.037875961512327194, 0.4778001010417938], [-0.7288646101951599, -0.16549959778785706, 0.033701784908771515], [0.8698932528495789, 0.11268221586942673, -0.3337450325489044], [-0.5489130616188049, -0.49730807542800903, 0.34982800483703613], [1.3919858932495117, -0.18031610548496246, -0.29819607734680176], [0.40162086486816406, -0.024776970967650414, -0.1733175367116928], [-0.9673424363136292, 0.2730551064014435, 0.7205177545547485], [0.2761675715446472, -0.21547573804855347, -0.22797323763370514], [0.8422470688819885, 0.28531840443611145, -0.6075484752655029], [-0.9649183750152588, -0.11343174427747726, 0.440578430891037], [0.20861981809139252, 0.3168995976448059, -0.3817315697669983], [0.13101595640182495, 0.24082304537296295, -0.4243215024471283], [0.011826667934656143, 0.2774938941001892, -0.2485351264476776], [0.5379206538200378, 0.018030840903520584, -0.309982568025589], [0.4333800971508026, 0.006292151287198067, 0.166518896818161], [0.9415228366851807, 0.1188533753156662, -0.6069473624229431], [0.7763894200325012, -0.23450377583503723, -0.24135175347328186], [-1.060146450996399, 0.10035523027181625, 0.3306023180484772], [0.10270140320062637, 0.046058498322963715, 0.05135668069124222], [-0.08405107259750366, 0.13206885755062103, 0.022067222744226456], [0.13201157748699188, -0.17006435990333557, 0.06257347017526627], [0.9216616153717041, 0.012000967748463154, -0.24478325247764587], [-0.7571732997894287, -0.19773393869400024, -0.07705090939998627], [0.7947220206260681, 0.09713825583457947, -0.27262866497039795], [-0.1319885551929474, 0.25601622462272644, -0.08387964218854904], [0.48420146107673645, 0.017933625727891922, 0.029798947274684906], [0.9813574552536011, 0.21506492793560028, -0.5167036652565002], [0.38523733615875244, -0.17290529608726501, -0.06720240414142609], [0.7826187014579773, 0.575408935546875, -0.5985714793205261], [0.20723378658294678, 0.3053975999355316, -0.059203509241342545], [-1.0756722688674927, 0.32134026288986206, 0.5617697238922119], [-0.5045967102050781, 0.04128303751349449, -0.13403676450252533], [0.7112886309623718, -0.39144858717918396, -0.45466122031211853], [-0.2904367744922638, 0.22841545939445496, -0.10381197184324265], [-0.7996273040771484, -0.0025426277425140142, 0.4121510684490204], [0.911174476146698, 0.10325875133275986, -0.6073696613311768], [-0.08634564280509949, 0.13614249229431152, -0.04086355119943619], [0.5076317191123962, -0.18434619903564453, 0.0028618767391890287], [0.4868471920490265, 0.6006191372871399, -0.9841108918190002], [0.6484277248382568, -0.24878087639808655, 0.08393359929323196], [0.21121564507484436, -0.1773412972688675, -0.2138885110616684], [-0.17901527881622314, 0.16888132691383362, -0.2168593406677246], [-1.0976855754852295, 0.2533611059188843, 0.08900471031665802], [0.23262715339660645, -0.1529867798089981, -0.2329089194536209], [0.8727642893791199, 0.08170054107904434, -0.44763943552970886], [0.03179101645946503, -0.3302839994430542, 0.375667542219162], [0.27456173300743103, 0.5383680462837219, -0.5268248915672302], [0.7819569706916809, -0.01816745661199093, -0.7604050636291504], [0.49412405490875244, -0.3732888400554657, -0.10228455811738968], [0.4133997857570648, -0.04652293026447296, -0.14841702580451965], [0.26268717646598816, -0.1657445728778839, 0.050803229212760925], [-0.6686924695968628, 0.18096813559532166, 0.3005216121673584], [0.7828854322433472, -0.07961315661668777, -0.6376771926879883], [0.448063462972641, 0.34485310316085815, -0.6355182528495789], [0.02504107356071472, 0.07326659560203552, -0.11894091963768005], [-0.6532619595527649, 0.15163910388946533, 0.38343894481658936], [-0.3164072334766388, 0.09805357456207275, -0.265411376953125], [-0.30403637886047363, -0.10010038316249847, 0.2943136990070343], [-0.5745318531990051, 0.14391562342643738, 0.21968746185302734], [0.23407770693302155, -0.31247806549072266, 0.19161826372146606], [-0.683194100856781, 0.034994032233953476, 0.01411763671785593], [-0.4813498258590698, -0.06068553030490875, 0.20258907973766327], [-0.22887477278709412, 0.41663795709609985, -0.24021823704242706], [-0.4822026789188385, -0.018559763208031654, 0.44761449098587036], [-0.16670021414756775, 0.2748940587043762, -0.23595039546489716], [0.33143413066864014, -0.04257141798734665, 0.15743955969810486], [-1.0043607950210571, 0.007391163147985935, 0.7166689038276672], [0.26208236813545227, -0.13751697540283203, -0.2693609893321991], [0.4829540550708771, -0.11307147145271301, 0.16301268339157104]]), new Matrix([[0.08026742935180664, 0.19998501241207123, -0.2491818368434906]]) ]) ]; 
        this.numLayers = this.layers.length;
    }
        
    predict(x){
        let output = parseInput(x);
        for ( let idx = 0; idx < this.numLayers ; idx++){
            output = this.layers[idx].predict(output);
        }
        return output;
    }
}

function parseInput(x){
    if ( x instanceof Matrix ){
        return new Matrix(x.mat);
    } else if ( ( x instanceof Array ) && x.length ){
        if ( x[0] instanceof Array ){
            return new Matrix( x );
        }else{
            return new Matrix( [x] );
        }
    } else {
        return null;
    }
}

// (2) code_apply_func

function applyFunc(matrix,fnc){
    const resMatrix = new Matrix(matrix.mat);
    for (let row = 0; row < matrix.rows; row++) {
        for (let col = 0; col < matrix.cols; col++) {
          resMatrix.mat[row][col] = fnc(matrix.mat[row][col]);
        }
    }
    return resMatrix;
}

// (3) code_act_funcs

softsign = (m) => applyFunc(m, (v) => v / (1+Math.abs(v)));
function softmax(m){
//     function fnc(v){
//         const expV = Math.exp(v-M
//         return 
//     }
// }

//     exp_x = np.exp(x - np.max(x))
//     return exp_x/exp_x.sum(axis=1, keepdims=True);

// (4) code_layers_class

class Dense {
  
    constructor(activation, weights) {
        this.activation = activation;
        [ this.kernel , this.bias ] = weights;
    }
    
    predict(x){
        return this.activation(x.dot(this.kernel).add(this.bias));
    }   
}

// (5) code_matrix

class Matrix {
  rows = null;
  cols = null;

  constructor(matrix) {
    this.rows = matrix.length;
    this.cols = matrix[0].length;
    this.mat = matrix;
    this.updateMinPad();
  }

  static Zeros(rows, cols) {
    const matrix = [];
    for (let row = 0; row < rows; row++) {
      matrix.push([...Array(cols).fill(0)])
    }
    return new Matrix(matrix);
  }

  shape() {
    return [this.rows, this.cols];
  }

  get(row = null, col = null) {
    row = (row == -1) ? this.rows - 1 : row;
    col = (col == -1) ? this.cols - 1 : col;
    if (col != null) {
      if (row != null) return this.mat[row][col];
      else {
        const column = [];
        for (let row = 0; row < this.rows; row++) {
          column.push([this.mat[row][col]]);
        }
        return new Matrix(column);
      }
    } else {
      if (row != null) return new Matrix([this.mat[row]]);
      else return new Matrix(this.mat);
    }
  }

  max() {
    let maxValue = -Infinity;
    let currValue;
    for (let row = 0; row < this.rows; row++) {
      currValue = Math.max(...this.mat[row]);
      if (currValue > maxValue) {
        maxValue = currValue;
      }
    }
    return maxValue;
  }

  min() {
    let minValue = Infinity;
    let currValue;
    for (let row = 0; row < this.rows; row++) {
      currValue = Math.min(...this.mat[row]);
      if (currValue < minValue) {
        minValue = currValue;
      }
    }
    return minValue;
  }

  round(dec = 1) {
    const OP = 10 ** dec;
    const newMatrix = Matrix.Zeros(this.rows, this.cols);
    for (let row = 0; row < this.rows; row++) {
      for (let col = 0; col < this.cols; col++) {
        newMatrix.mat[row][col] = Math.round(this.mat[row][col] * OP) / OP;
      }
    }
    newMatrix.updateMinPad();
    return newMatrix;
  }

  updateMinPad() {
    const flatMatrix = this.mat.reduce((acc, x) => acc.concat(x), []);
    const idxLength = flatMatrix.map((x, idx) => [String(x).length, idx]);
    idxLength.sort();
    this.minPad = idxLength.at(-1)[0];
  }

  print() {
    this.updateMinPad();
    let result = "";
    const addPad = (x) => String(x).padStart(this.minPad, " ");
    for (let row = 0; row < this.rows; row++) {
      result += `[ ${this.mat[row].map(addPad).join(" , ")} ]\n`;
    }
    console.log(result);
  }

  addValue(value) {
    if (typeof (value) !== "number") return null;
    const newMatrix = Matrix.Zeros(this.rows, this.cols);
    for (let row = 0; row < this.rows; row++) {
      for (let col = 0; col < this.cols; col++) {
        newMatrix.mat[row][col] = this.mat[row][col] + value;
      }
    }
    return newMatrix;
  }

  add(matrix) {
    if (typeof (matrix) === "number") return this.addValue(matrix);

    if (!matrix instanceof Matrix) return null;
    
    const newMatrix = Matrix.Zeros(this.rows, this.cols);

    if (this.eqRows(matrix) && this.eqCols(matrix)) {
      for (let row = 0; row < this.rows; row++) {
        for (let col = 0; col < matrix.cols; col++) {
          newMatrix.mat[row][col] = this.mat[row][col] + matrix.mat[row][col];
        }
      }
    } else if (this.eqRows(matrix) && (matrix.cols === 1)) {

      for (let row = 0; row < this.rows; row++) {
        for (let col = 0; col < matrix.cols; col++) {
          newMatrix.mat[row][col] = this.mat[row][col] + matrix.mat[row][0];
        }
      }
    } else if (this.eqCols(matrix) && (matrix.rows === 1)) {
      for (let row = 0; row < this.rows; row++) {
        for (let col = 0; col < matrix.cols; col++) {
          newMatrix.mat[row][col] = this.mat[row][col] + matrix.mat[0][col];
        }
      }
    } else {
      return null;
    }
    return newMatrix;
  }

  multiplyValue(value) {
    if (typeof (value) !== "number") return null;
    const newMatrix = Matrix.Zeros(this.rows, this.cols);
    for (let row = 0; row < this.rows; row++) {
      for (let col = 0; col < this.cols; col++) {
        newMatrix.mat[row][col] = this.mat[row][col] * value;
      }
    }
    return newMatrix;
  }

  multiply(matrix) {

    const newMatrix = Matrix.Zeros(this.rows, matrix.cols);
    if (typeof (matrix) === "number") return this.multiplyValue(matrix);

    if (!matrix instanceof Matrix) return null;

    if (this.eqRows(matrix) && this.eqCols(matrix)) {
      for (let row = 0; row < this.rows; row++) {
        for (let col = 0; col < matrix.cols; col++) {
          newMatrix.mat[row][col] = this.mat[row][col] * matrix.mat[row][col];
        }
      }
    } else if (this.eqRows(matrix) && (matrix.cols === 1)) {

      for (let row = 0; row < this.rows; row++) {
        for (let col = 0; col < matrix.cols; col++) {
          newMatrix.mat[row][col] = this.mat[row][col] * matrix.mat[row][0];
        }
      }
    } else if (this.eqCols(matrix) && (matrix.rows === 1)) {
      for (let row = 0; row < this.rows; row++) {
        for (let col = 0; col < matrix.cols; col++) {
          newMatrix.mat[row][col] = this.mat[row][col] * matrix.mat[0][col];
        }
      }
    } else {
      return null;
    }

    return newMatrix;
  }

  dot(matrix) {
    if (this.cols !== matrix.rows) return null;
    const REPS = this.cols;
    let value;
    const newMatrix = Matrix.Zeros(this.rows, matrix.cols);
    for (let row = 0; row < this.rows; row++) {
      for (let col = 0; col < matrix.cols; col++) {
        value = 0;
        for (let rep = 0; rep < REPS; rep++) {
          value += this.mat[row][rep] * matrix.mat[rep][col];
        }
        newMatrix.mat[row][col] = value;
      }
    }
    return newMatrix;
  }

  eqRows(matrix) {
    return this.rows === matrix.rows;
  }

  eqCols(matrix) {
    return this.cols === matrix.cols;
  }
}

// XOR

const datasets = {
    xor : new Matrix([[0,0],[0,1],[1,0],[1,1]]),
    iris : new Matrix([
            [0.39, 0.38, 0.54, 0.50],
            [0.11, 0.50, 0.10, 0.04],
            [0.61, 0.33, 0.61, 0.58]
        ])
};