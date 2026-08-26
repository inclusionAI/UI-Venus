import subprocess
import json

# Complete application mapping.
APP_MAPPING = {
    "小红书": "com.xingin.xhs/com.xingin.xhs.index.v2.IndexActivityV2",
    "微博": "com.sina.weibo/com.sina.weibo.MainTabActivity",
    "知乎": "com.zhihu.android/com.zhihu.android.app.ui.activity.MainActivity",
    "豆瓣": "com.douban.frodo/com.douban.frodo.activity.SplashActivity",
    "拼多多": "com.xunmeng.pinduoduo/com.xunmeng.pinduoduo.ui.activity.HomeActivity",
    "唯品会": "com.achievo.vipshop/com.achievo.vipshop.activity.LodingActivity",
    "支付宝": "com.eg.android.AlipayGphone/com.eg.android.AlipayGphone.AlipayLogin",
    "百度地图": "com.baidu.BaiduMap/com.baidu.baidumaps.WelcomeScreen",
    "美团": "com.sankuai.meituan/com.meituan.android.pt.homepage.activity.MainActivity",
    "美团外卖": "com.sankuai.meituan.takeoutmanage/com.sankuai.meituan.takeoutmanage.ui.MainActivity",
    "饿了么": "me.ele/me.ele.Launcher",
    "大众点评": "com.dianping.v1/com.dianping.main.guide.SplashScreenActivity",
    "携程旅行": "ctrip.android.view/ctrip.android.publicproduct.home.business.activity.CtripHomeActivity",
    "同程": "com.tongcheng.android/com.tongcheng.android.host.HomeActivity",
    "铁路12306": "com.MobileTicket/com.MobileTicket.ui.activity.MainActivity",
    "滴滴出行": "com.sdu.didi.psnger/com.didi.sdk.app.MainActivity",
    "58同城": "com.wuba/com.wuba.home.activity.HomeActivity",
    "哔哩哔哩": "tv.danmaku.bili/tv.danmaku.bili.MainActivityV2",
    "快手": "com.smile.gifmaker/com.yxcorp.gifshow.HomeActivity",
    "腾讯视频": "com.tencent.qqlive/com.tencent.qqlive.ona.activity.SplashHomeActivity",
    "爱奇艺": "com.qiyi.video/org.qiyi.android.video.MainActivity",
    "优酷视频": "com.youku.phone/com.youku.phone.ActivityWelcome",
    "芒果TV": "com.hunantv.imgo.activity/com.hunantv.imgo.activity.MainActivity",
    "QQ音乐": "com.tencent.qqmusic/com.tencent.qqmusic.activity.AppStarterActivity",
    "酷我音乐": "cn.kuwo.player/cn.kuwo.player.activities.EntryActivity",
    "喜马拉雅": "com.ximalaya.ting.android/com.ximalaya.ting.android.host.activity.MainActivity",
    "汽水音乐": "com.luna.music/com.luna.biz.main.main.MainActivity",
    "蜻蜓FM": "fm.qingting.qtradio/fm.qingting.qtradio.TabHostActivity",
    "今日头条": "com.ss.android.article.news/com.ss.android.article.news.activity.MainActivity",
    "番茄免费小说": "com.dragon.read/com.dragon.read.pages.main.MainFragmentActivity",
    "七猫免费小说": "com.kmxs.reader/com.kmxs.reader.home.ui.HomeActivity",
    "WPS": "cn.wps.moffice_eng/cn.wps.moffice.documentmanager.PreStartActivity",
    "飞书": "com.ss.android.lark/com.ss.android.lark.main.MainActivity",
    "元宝": "com.tencent.yuanbao/com.tencent.yuanbao.main.MainActivity",
    "豆包": "com.larus.nova/com.larus.nova.main.MainActivity",
    "千问": "com.alibaba.qwen/com.alibaba.qwen.main.MainActivity",
    "贝壳找房": "com.lianjia.beike/com.lianjia.activity.MainActivity",
    "安居客": "com.anjuke.android.app/com.anjuke.android.app.main.MainActivity",
    "Markor": "net.gsantner.markor/net.gsantner.markor.activity.MainActivity",
    "星穹铁道": "com.miHoYo.hkrpg/com.miHoYo.hkrpg.MainActivity",
    "同花顺": "com.hexin.plat.android/com.hexin.plat.android.Hexin"
}

def run_adb_cmd(cmd):
    """Run an ADB command and return its output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def verify():
        # Select the first connected device automatically.
    device_list = run_adb_cmd("adb devices")
    lines = device_list.split('\n')[1:]
    devices = [line.split('\t')[0] for line in lines if line.strip() and 'device' in line]
    
    if not devices:
        print("❌ 未发现连接的设备，请检查 ADB 连接！")
        return
    
    device_id = devices[0]
    print(f"🚀 开始在设备 [{device_id}] 上验证 {len(APP_MAPPING)} 个应用...\n")
    print(f"{'应用名称':<12} | {'验证结果':<10} | {'备注'}")
    print("-" * 60)

    for app_name, mapping in APP_MAPPING.items():
        package = mapping.split('/')[0]
        
    # 1. Verify that the package exists.
        check_pkg = run_adb_cmd(f"adb -s {device_id} shell pm list packages {package}")
        if package not in check_pkg:
            print(f"{app_name:<12} | ❌ 失败      | 包名未安装: {package}")
            continue
        
    # 2. Verify the activity path.
    # pm resolve-activity prints package=... details when the path is valid.
        check_act = run_adb_cmd(f"adb -s {device_id} shell pm resolve-activity -n {mapping}")
        
        if "package=" in check_act:
            print(f"{app_name:<12} | ✅ 成功      | 映射正确")
        else:
    # 3. If validation fails, query the system for the recommended entry point.
            suggest = run_adb_cmd(f"adb -s {device_id} shell \"cmd package resolve-activity --brief {package} | tail -n 1\"")
            print(f"{app_name:<12} | ❌ 路径错误  | 建议: {suggest}")

if __name__ == "__main__":
    verify()
