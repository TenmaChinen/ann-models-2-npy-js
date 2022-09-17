class Model{
    constructor(){
        // (1) code_layers_class_instance
        this.layers = [ Dense(activation=softsign, weights=[ new Matrix([[0.11260133981704712, -0.06263938546180725, 0.15319037437438965, 0.020601341500878334, -0.0007953237509354949, -0.23051193356513977, 0.14036139845848083, -0.11829246580600739, -0.07141757011413574, 0.1388012021780014, 0.1635434776544571, -0.03836258128285408, -0.20044288039207458, 0.007601344026625156, 0.16347193717956543, 0.18015435338020325, -0.13234974443912506, 0.009112890809774399, -0.29848358035087585, 0.07832381874322891, -0.3636835217475891, 0.39837440848350525, -0.1697680652141571, 0.050392668694257736, -0.3037853240966797, -0.028046678751707077, 0.20863446593284607, -0.40443193912506104, 0.22383354604244232, 0.1359354853630066, 0.1546539068222046, 0.0448286235332489, 0.32631629705429077, 0.23531340062618256, -0.05736469849944115, 0.2749670743942261, -0.3088006377220154, -0.2069418579339981, -0.10967753827571869, -0.054829686880111694, 0.06798169761896133, 0.17552000284194946, -0.09539303183555603, -0.17424850165843964, -0.012446166947484016, -0.34212666749954224, 0.2733827531337738, -0.3534102737903595, -0.12530121207237244, -0.06331346184015274, -0.15060773491859436, -0.5018891096115112, -0.020021390169858932, -0.4127437174320221, 0.04862995445728302, -0.13628463447093964, -0.2286730259656906, 0.24207034707069397, 0.050511911511421204, 0.02703571692109108, 0.21409057080745697, -0.4637415111064911, 0.2838734984397888, -0.2747940421104431, -0.15878209471702576, -0.18176496028900146, -0.12602941691875458, -0.11457373946905136, 0.41330525279045105, -0.2910632789134979, 0.23155951499938965, -0.3152949810028076, 0.016355572268366814, 0.24996596574783325, 0.22975146770477295, -0.24841468036174774, 0.1610110104084015, -0.21809926629066467, -0.2914019525051117, -0.025098254904150963, -0.32175466418266296, -0.22203867137432098, 0.04223157837986946, -0.45789700746536255, 0.1601463407278061, -0.044761739671230316, -0.12765581905841827, -0.12687020003795624, -0.33744123578071594, 0.0916023850440979, -0.14902837574481964, -0.13101568818092346, -0.2921641170978546, -0.012952168472111225, -0.12039286643266678, 0.1322234570980072, 0.24413494765758514, -0.11125920712947845, 0.26488399505615234, -0.18334195017814636], [-0.20300303399562836, -0.20250999927520752, -0.2393885999917984, 0.3649347126483917, -0.18255172669887543, 0.5670803785324097, -0.4771062433719635, -0.20151403546333313, -0.3369658887386322, -0.31678664684295654, -0.2263667732477188, 0.15672826766967773, 0.66778165102005, -0.4396800398826599, -0.4680306613445282, 0.3281704783439636, 0.20204664766788483, 0.38820570707321167, 0.6113299131393433, -0.23925083875656128, 0.627421498298645, -0.6058356761932373, -0.27484366297721863, 0.33139869570732117, 0.6573659777641296, 0.297420471906662, 0.03695713356137276, 0.7488652467727661, 0.3862823247909546, -0.3165923058986664, -0.18515510857105255, -0.04756671190261841, -0.4430643916130066, 0.05933961644768715, 0.481926292181015, -0.2594273090362549, 0.596179723739624, 0.6480512619018555, 0.3428886830806732, 0.10362184792757034, 0.3053765296936035, 0.05824857950210571, -0.3261723220348358, -0.0087257856503129, -0.2878294885158539, 0.7235475778579712, -0.31887540221214294, 0.4486827850341797, 0.30794575810432434, -0.19817477464675903, 0.34475961327552795, 0.7223930954933167, 0.35352054238319397, 0.7167814373970032, -0.08596372604370117, 0.015574513003230095, 0.09127405285835266, -0.734009861946106, -0.4970184564590454, -0.17992472648620605, 0.04347618669271469, 0.2168305516242981, -0.535094678401947, 0.233730286359787, 0.6149948239326477, 0.10374578833580017, -0.13038040697574615, -0.1564231961965561, 0.17553631961345673, -0.18925771117210388, -0.31875261664390564, 0.3894185721874237, 0.25707054138183594, -0.5197805762290955, -0.7115519642829895, 0.6680719256401062, -0.3352620005607605, 0.5352780818939209, 0.6860467791557312, -0.12054997682571411, 0.4823354482650757, 0.49832698702812195, -0.5642139911651611, 0.08513807505369186, -0.4167064428329468, 0.5422559976577759, 0.5105577707290649, 0.5483770370483398, 0.6181320548057556, -0.27856478095054626, -0.2303207367658615, 0.6908460259437561, 0.4788833558559418, -0.07208488881587982, 0.14649665355682373, -0.5023928880691528, -0.17147663235664368, 0.11857499927282333, -0.04488369822502136, 0.4139120280742645], [0.46793919801712036, 0.46650996804237366, 0.18820685148239136, -0.4611261487007141, 0.04891188442707062, -0.40662479400634766, 0.2587321698665619, 0.3744438886642456, 0.34277886152267456, 0.20519192516803741, 0.005512824282050133, 0.13870014250278473, -0.5439989566802979, 0.32464003562927246, 0.35709601640701294, -0.3152191638946533, -0.39762887358665466, -0.1308290958404541, -0.5525469779968262, -0.03400645777583122, -0.7946569323539734, 0.35102659463882446, -0.0722762793302536, -0.4506736695766449, -0.433085560798645, 0.008861575275659561, 0.08615139126777649, -0.7974079847335815, -0.1096305251121521, 0.4692235291004181, 0.10305806994438171, 0.06808649748563766, 0.6369752287864685, 0.04183953255414963, -0.3944716155529022, 0.2401801347732544, -0.6803700923919678, -0.21605099737644196, -0.20727407932281494, -0.15096578001976013, 0.22358275949954987, -0.23282431066036224, 0.08714664727449417, -0.30984970927238464, 0.3174264132976532, -0.807585597038269, 0.5553763508796692, -0.5465638041496277, -0.5430338978767395, 0.408031165599823, -0.12686412036418915, -0.5572149753570557, -0.295998752117157, -0.4903077781200409, 0.1366611123085022, 0.07027935981750488, -0.17107202112674713, 0.5556828379631042, 0.6039367914199829, 0.5206161141395569, 0.1891987919807434, -0.13205857574939728, 0.34685608744621277, -0.424306184053421, -0.31991440057754517, -0.039451420307159424, -0.0019070011330768466, 0.09890087693929672, 0.16400690376758575, -0.35560253262519836, 0.1046563982963562, -0.5172900557518005, -0.46295154094696045, 0.18970656394958496, 0.5883646011352539, -0.5712882876396179, 0.14594975113868713, -0.45266392827033997, -0.6035924553871155, 0.2716098725795746, -0.18496960401535034, -0.6349552869796753, 0.26940998435020447, -0.3923710286617279, 0.4870300590991974, -0.5461212396621704, -0.6554077863693237, -0.271039754152298, -0.4957695007324219, 0.34096983075141907, 0.07259219884872437, -0.5119088292121887, -0.09714263677597046, -0.3077101409435272, -0.12685668468475342, 0.0578472726047039, 0.005319959484040737, 0.09136462211608887, 0.0018125185742974281, -0.5542053580284119], [0.560455858707428, 0.6585878729820251, 0.7133481502532959, -0.7255722284317017, -0.08804059028625488, 0.18004833161830902, 0.19755259156227112, 0.6762392520904541, 0.44284433126449585, -0.4253097176551819, 0.41770845651626587, 0.24536029994487762, -0.1964288055896759, 0.7038269639015198, 0.5358214378356934, -0.24370576441287994, -0.5694323182106018, -0.4742044508457184, -0.5925307869911194, 0.0708337277173996, -0.2668760418891907, 0.5730328559875488, 0.404353529214859, -0.2728331983089447, -0.7379643321037292, -0.02272048033773899, 0.23769401013851166, -0.7405992150306702, 0.16964305937290192, 0.3674246668815613, 0.19869542121887207, -0.2639964818954468, 0.3469027578830719, 0.06544137746095657, -0.012964456342160702, 0.5003829598426819, -0.27871471643447876, -0.6653006076812744, 0.053133074194192886, -0.6324912309646606, 0.1378721296787262, -0.20798315107822418, 0.1764308512210846, -0.5541253685951233, 0.6707211136817932, -0.6399324536323547, 0.44344380497932434, -0.02859347313642502, -0.34085023403167725, 0.16136209666728973, -0.42569631338119507, -0.5169292688369751, -0.61680668592453, -0.3793170750141144, 0.5810291171073914, -0.1981697529554367, -0.5411140322685242, 0.6087235808372498, 0.602630615234375, 0.5105482339859009, 0.37582018971443176, -0.33464717864990234, 0.7468266487121582, -0.4671598970890045, -0.3196515440940857, -0.225265771150589, 0.24165105819702148, -0.06857013702392578, 0.25185585021972656, -0.4322403371334076, 0.7418228387832642, -0.4793645143508911, -0.5221044421195984, 0.24025675654411316, 0.6979892253875732, -0.5869821906089783, 0.6831929087638855, -0.4807907044887543, -0.19874171912670135, 0.7047222256660461, -0.6370059847831726, -0.6327422261238098, 0.027714964002370834, -0.07024512439966202, 0.6377715468406677, -0.27252644300460815, -0.5969975590705872, -0.6453425884246826, -0.5048813819885254, 0.6057981848716736, -0.16571550071239471, -0.5104151368141174, -0.2007659375667572, -0.21226327121257782, -0.6445598006248474, -0.19783395528793335, 0.2319445013999939, 0.050532300025224686, 0.5790830254554749, -0.37587839365005493]]), new Matrix([-0.22013670206069946, -0.5688004493713379, -0.5221417546272278, 0.5773018002510071, -0.37560588121414185, -0.2026003897190094, 0.012274617329239845, -0.4371994137763977, -0.30834096670150757, 0.23922760784626007, -0.31196531653404236, -0.41033366322517395, -0.06053964048624039, -0.24014650285243988, -0.44710874557495117, 0.010531620122492313, 0.39652615785598755, 0.20472174882888794, 0.15208019316196442, 0.17846207320690155, 0.08021097630262375, -0.18548358976840973, -0.51976478099823, -0.037797775119543076, 0.16063672304153442, 0.25656571984291077, -0.40770572423934937, 0.1696833223104477, -0.1201365739107132, -0.4185276925563812, -0.13830578327178955, 0.3853749632835388, -0.4448392987251282, -0.2149745523929596, 0.09908848255872726, -0.48126286268234253, -0.09920761734247208, 0.24273961782455444, -0.2620018720626831, 0.4111785888671875, 0.1909540295600891, 0.2004472315311432, -0.034445859491825104, -0.11888350546360016, -0.42522215843200684, -0.035735420882701874, -0.6215574741363525, -0.07954390347003937, 0.472412109375, 0.4409647583961487, 0.29385218024253845, 0.20673948526382446, 0.4252619743347168, -0.07279520481824875, -0.1380937695503235, 0.507254421710968, 0.4202617406845093, -0.4774991571903229, -0.5562546849250793, -0.5194611549377441, -0.4091033339500427, 0.3994235694408417, -0.299694687128067, -0.10465638339519501, 0.03098229505121708, 0.42332738637924194, -0.4883083999156952, 0.4589031934738159, 0.28084245324134827, 0.02741892822086811, -0.5040648579597473, 0.6123883128166199, 0.4931405186653137, 0.0879860669374466, -0.6129640340805054, -0.028785964474081993, -0.4556107521057129, 0.4541674554347992, 0.05054395645856857, -0.5057275295257568, 0.35642677545547485, 0.33984437584877014, -0.03468577563762665, -0.16844911873340607, -0.6225302219390869, 0.007574617862701416, 0.3694070279598236, 0.38177743554115295, 0.5006250739097595, -0.5106339454650879, -0.031222525984048843, 0.07401924580335617, 0.1782495230436325, -0.1533532440662384, 0.433560311794281, 0.07012811303138733, -0.4499131441116333, 0.417861670255661, -0.41724035143852234, 0.34752267599105835]) ]), Dense(activation=softmax, weights=[ new Matrix([[-0.7045841217041016, 0.1576550006866455, 0.3307724595069885], [-0.6933373808860779, -0.36005985736846924, 0.8358954787254333], [-0.6370638608932495, -0.3363899290561676, 0.8813869953155518], [0.7073062062263489, 0.22859223186969757, -0.9688439965248108], [0.3169967532157898, -0.020795824006199837, 0.07135160267353058], [0.35340753197669983, -0.423319935798645, 0.041703276336193085], [-0.7644215226173401, 0.1625000685453415, 0.07217387855052948], [-0.4949248433113098, -0.426778644323349, 0.3091317415237427], [-0.5254310965538025, 0.12458823621273041, 0.4856657385826111], [0.06785270571708679, 0.44630345702171326, -0.05064946040511131], [-0.5515469908714294, 0.01100878231227398, 0.16358591616153717], [0.16057904064655304, -0.46533337235450745, -0.15700717270374298], [0.8750700950622559, -0.31752580404281616, -0.143421933054924], [-1.2433173656463623, 0.06621500104665756, 0.31608837842941284], [-0.735257625579834, -0.0009999147150665522, 0.6653892397880554], [0.6997905969619751, -0.14708347618579865, -0.1058201715350151], [0.5978288650512695, 0.11566375195980072, -0.3427601754665375], [0.9829527735710144, 0.09479537606239319, -0.10222893208265305], [1.2975629568099976, -0.2693994343280792, -0.35114333033561707], [-0.4085438549518585, 0.21742023527622223, 0.07173715531826019], [1.020729422569275, -0.3574165105819702, -0.3362981379032135], [-1.2883459329605103, 0.07956423610448837, 0.18788495659828186], [-0.3181471526622772, -0.3135736286640167, 0.3103610575199127], [0.7091710567474365, -0.09113016724586487, -0.02450118400156498], [0.9711574912071228, -0.20687659084796906, -0.35796675086021423], [0.19141992926597595, 0.08093320578336716, 0.05455873906612396], [0.3867396414279938, -0.14164918661117554, 0.09789647161960602], [0.9791228771209717, -0.4506133496761322, -0.5042333602905273], [0.06050311401486397, -0.07606492936611176, 0.11529962718486786], [-0.901419997215271, 0.06646852940320969, 0.7087863087654114], [-0.3520043194293976, 0.25699755549430847, 0.319193571805954], [-0.247444286942482, 0.14734295010566711, -0.22388482093811035], [-1.0193742513656616, 0.12456128001213074, 0.5415586233139038], [-0.31995704770088196, 0.042254481464624405, 0.09243908524513245], [0.377570778131485, -0.1406511664390564, -0.18591639399528503], [-0.43286043405532837, 0.012011795304715633, 0.8074983358383179], [1.1241027116775513, -0.23379501700401306, -0.0831977128982544], [1.1073557138442993, -0.06261327117681503, -0.23565536737442017], [0.45878851413726807, -0.3616761565208435, -0.12259598821401596], [0.5571314096450806, 0.2472495436668396, -0.574765145778656], [-0.009854950942099094, 0.14042620360851288, 0.2653825879096985], [0.47117453813552856, 0.03528435155749321, -0.09444637596607208], [-0.4261932969093323, -0.0297820083796978, 0.014029588550329208], [-0.036288514733314514, -0.11490228027105331, -0.35663408041000366], [-0.23113226890563965, -0.15863998234272003, 0.6460028290748596], [0.89857017993927, -0.478819340467453, -0.3862403631210327], [-0.49995699524879456, -0.13815510272979736, 0.6764475703239441], [0.8358317017555237, -0.17463995516300201, 0.00857518706470728], [0.29185187816619873, 0.032399632036685944, -0.4826720356941223], [-0.5861974954605103, 0.2245735377073288, 0.003676121588796377], [0.7797263860702515, 0.11146192997694016, -0.09204085171222687], [0.9923403859138489, -0.14557777345180511, -0.2612467110157013], [0.793514609336853, 0.16958996653556824, -0.43938660621643066], [1.1094069480895996, -0.28542202711105347, -0.15904086828231812], [-0.1814749538898468, 0.10332132130861282, 0.3951393961906433], [-0.39480599761009216, 0.34871789813041687, -0.031702376902103424], [0.2772209942340851, -0.06038632243871689, -0.5360686182975769], [-0.7020698189735413, 0.07437264919281006, 0.5022850036621094], [-0.9266869425773621, 0.09408702701330185, 0.8602709174156189], [-0.3271732032299042, -0.3196667730808258, 0.5254281759262085], [-0.11890383809804916, -0.22709842026233673, 0.34581464529037476], [-0.1150098666548729, 0.14522992074489594, -0.09059353172779083], [-1.1167782545089722, 0.10310167819261551, 0.4129412770271301], [0.540450930595398, -0.3781037926673889, -0.18725521862506866], [0.9178759455680847, -0.19113591313362122, -0.14925619959831238], [-0.019318057224154472, 0.2579286992549896, 0.006889176554977894], [0.22821436822414398, -0.3391318619251251, 0.03255993127822876], [-0.5125656127929688, 0.2439759075641632, -0.001567827770486474], [-0.19116610288619995, 0.1502087563276291, 0.2580397427082062], [0.23055951297283173, 0.19616170227527618, -0.22345978021621704], [-0.514155387878418, -0.10481051355600357, 0.9200977087020874], [0.5794602036476135, 0.21648725867271423, -0.5735408067703247], [0.8249086141586304, 0.20573830604553223, -0.6085161566734314], [-0.6405943632125854, 0.2166007161140442, 0.07956632971763611], [-0.6809245944023132, 0.11354079842567444, 0.6720460057258606], [1.0827287435531616, -0.3318155109882355, -0.27228811383247375], [-0.7871648073196411, -0.009332568384706974, 0.6663987040519714], [0.7922017574310303, -0.13676251471042633, -0.757256269454956], [0.849775493144989, -0.1430421769618988, -0.030916288495063782], [-0.5045711398124695, -0.42533886432647705, 0.8794710040092468], [1.0897045135498047, -0.09259343892335892, -0.4208913743495941], [0.923745334148407, -0.18016065657138824, -0.5101954340934753], [-0.10470271855592728, 0.04997436702251434, 0.11167969554662704], [0.005041580181568861, -0.22212378680706024, -0.285521000623703], [-0.7291004061698914, -0.19187059998512268, 0.6677528619766235], [0.9751729369163513, -0.1497431844472885, -0.07225272059440613], [0.8867303729057312, -0.25120848417282104, -0.5477281808853149], [0.6269422173500061, -0.1620943397283554, -0.6283140182495117], [0.7115569710731506, -0.2297438383102417, -0.7866690158843994], [-0.6798187494277954, -0.21775369346141815, 0.7629867196083069], [-0.46236032247543335, -0.010751908645033836, -0.1469077318906784], [0.9771138429641724, -0.336826354265213, -0.3283936083316803], [0.23377946019172668, -0.051696404814720154, -0.13120466470718384], [-0.21296212077140808, -0.13176442682743073, -0.24675390124320984], [0.37973979115486145, 0.313178688287735, -0.415355384349823], [-0.4444963037967682, 0.07908621430397034, -0.0018463358283042908], [0.1795479953289032, -0.28854721784591675, -0.012320232577621937], [-0.11412748694419861, -0.01848333328962326, -0.06557048857212067], [-0.33381301164627075, -0.44671520590782166, 0.5762463808059692], [1.068068027496338, -0.09339277446269989, -0.3805347681045532]]), new Matrix([0.07611583918333054, 0.3162042498588562, -0.4474796652793884]) ]) ]; 
        this.numLayers = this.layers.length;
    }
        
    pred(x){
        let output = parseInput(x);
        for ( let idx = 0, idx < this.numLayers ; idx++){
            output = this.layers[idx];
        }
        return output;
    }
}

function parseInput(x){
    if ( x instanceof Matrix ){
        return new Matrix( x.mat );
    } else if ( x instanceof Array ){
        return  x[0].length ? x : [x] ;
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
softsign = (m) => applyFunc(m, (v) => v / (1+Math.abs(v) );
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
  
    constructor(activation, weights ) {
        this.activation = activation;
        [ this.kernel , this.bias ] = weights;
    }
    
    predict(x){
        return this.activation(x.dot(this.kernel).add(this.bias);
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

  shape(){
    return [this.rows,this.cols];
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

  round(dec=1){
    const OP = 10 ** dec;
    const newMatrix = Matrix.Zeros(this.rows, this.cols);
    for (let row = 0; row < this.rows; row++) {
      for (let col = 0; col < this.cols; col++) {
        newMatrix.mat[row][col] = Math.round(this.mat[row][col]*OP)/OP;
      }
    }
    newMatrix.updateMinPad();
    return newMatrix;
  }

  updateMinPad() {
    const flatMatrix = this.mat.reduce( (acc,x) => acc.concat(x), [] );
    const idxLength = flatMatrix.map( (x,idx) => [idx, String(x).length]);
    idxLength.sort();
    this.minPad = idxLength.at(-1)[1];
  }

  print() {
    let result = "";
    const addPad = (x) => String(x).padStart(this.minPad, " ");
    for (let row = 0; row < this.rows; row++) {
      result += `[ ${this.mat[row].map(addPad).join(" , ")} ]\n`;
    }
    console.log(result);
  }

  addValue(num) {
    for (let row = 0; row < this.rows; row++) {
      for (let col = 0; col < this.cols; col++) {
        this.mat[row][col] += num;
      }
    }
    return this;
  }

  add(matrix) {
    if (this.cols !== matrix.cols) return null;
    if (matrix.rows === 1){
      const repMat = [];
      for(let idx=0; idx < this.rows; idx++) repMat.push([...matrix.mat[0]]);
      matrix = new Matrix(repMat);
    }
    const newMatrix = Matrix.Zeros(this.rows, matrix.cols);
    for (let row = 0; row < this.rows; row++) {
      for (let col = 0; col < matrix.cols; col++) {
        newMatrix.mat[row][col] = this.mat[row][col] + matrix.mat[row][col];
      }
    }
    newMatrix.updateMinPad();
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
    newMatrix.updateMinPad();
    return newMatrix;
  }
}

const matA = new Matrix([[0, 2, 3], [4, 5, 6]]);
const matB = new Matrix([[10, 2], [4, 5], [2, 1]]);
const matC = new Matrix([[4, 1, 2], [6, 3, 0]]); 