#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/point-to-point-net-device.h"
#include "ns3/applications-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/tcp-socket-base.h"
#include "gym_tcp_env.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("Topo3ParkingLot");

namespace
{

// -----------------------------------------------------------------------------
// 실험 공통 상수 및 결과 저장 경로
// control interval, simulation time, packet size, bottleneck을 설정하고 결과 파일 저장 위치를 정의합니다.
// -----------------------------------------------------------------------------

constexpr double STEP_TIME = 0.1;
constexpr double SIM_TIME = 160.0;
constexpr uint32_t PACKET_SIZE = 536;

const std::string RESULT_DIR = "scratch/results/topo3_rtt/";
const std::string BOTTLENECK_RATE = "15Mbps";
const std::string QUEUE_SIZE = "150p";


// -----------------------------------------------------------------------------
// 결과 로그 파일 stream
// S1/S2의 cwnd, RTT, goodput 출력이 baseline/RL mode별 txt 파일로 저장되게 지정합니다.
// plot.py와 analyze_intervals.py는 이 파일들을 읽어 결과를 분석합니다.
// -----------------------------------------------------------------------------

std::ofstream f_cwnd_s1;
std::ofstream f_cwnd_s2;
std::ofstream f_rtt_s1;
std::ofstream f_rtt_s2;
std::ofstream f_goodput_s1;
std::ofstream f_goodput_s2;


// -----------------------------------------------------------------------------
// Flow별 수신 byte 누적 배열
// control interval 동안 sink에서 수신한 byte를 flow별로 누적합니다.
//
// flowId:
//   1 = S1 FTP/background flow
//   2 = S2 primary video-like protected flow
//   3 = S3 burst cross traffic
// -----------------------------------------------------------------------------

std::array<uint64_t, 4> g_rxBytes = {0, 0, 0, 0};


// -----------------------------------------------------------------------------
// 결과 파일 열기
// mode에 따라 baseline_*.txt 또는 rl_*.txt 파일을 생성합니다.
// prefix는 main()에서 mode 값을 기준으로 baseline 또는 rl로 결정됩니다.
// -----------------------------------------------------------------------------

void OpenResultFiles(const std::string &prefix)
{
    std::filesystem::create_directories(RESULT_DIR);

    f_cwnd_s1.open(RESULT_DIR + prefix + "_cwnd_s1.txt");
    f_cwnd_s2.open(RESULT_DIR + prefix + "_cwnd_s2.txt");
    f_rtt_s1.open(RESULT_DIR + prefix + "_rtt_s1.txt");
    f_rtt_s2.open(RESULT_DIR + prefix + "_rtt_s2.txt");
    f_goodput_s1.open(RESULT_DIR + prefix + "_goodput_s1.txt");
    f_goodput_s2.open(RESULT_DIR + prefix + "_goodput_s2.txt");
}


// -----------------------------------------------------------------------------
// 결과 파일 닫기
// simulation 종료 후 열려 있는 로그 파일 stream을 정리합니다.
// -----------------------------------------------------------------------------

void CloseResultFiles()
{
    f_cwnd_s1.close();
    f_cwnd_s2.close();
    f_rtt_s1.close();
    f_rtt_s2.close();
    f_goodput_s1.close();
    f_goodput_s2.close();
}


// -----------------------------------------------------------------------------
// ns3-gym step 호출
// 여기서는 0.1초마다 GymTcpEnv::Notify()를 호출해 Python agent와 observation/action interaction을 수행합니다.
// -----------------------------------------------------------------------------

void GymStep(Ptr<GymTcpEnv> env)
{
    if (Simulator::IsFinished())
    {
        return;
    }

    env->Notify();
    Simulator::Schedule(Seconds(STEP_TIME), &GymStep, env);
}


// -----------------------------------------------------------------------------
// PacketSink 수신 callback
// 각 destination sink가 packet을 수신할 때마다 flow별 byte를 누적합니다.
// GymTcpEnv에는 ReportRx()를 통해 S1/S2 observation에 사용할 수신 byte를 전달합니다.
// -----------------------------------------------------------------------------

void RxCallback(
    Ptr<GymTcpEnv> env,
    uint32_t flowId,
    Ptr<const Packet> packet,
    const Address &address)
{
    uint32_t bytes = packet->GetSize();

    env->ReportRx(flowId, bytes);

    if (flowId < g_rxBytes.size())
    {
        g_rxBytes[flowId] += bytes;
    }
}


// -----------------------------------------------------------------------------
// Queue drop callback
// 여기서는 bottleneck queue에서 drop이 발생하면 GymTcpEnv에 loss signal을 전달합니다.
// 구현 방식에서 flow별 loss attribution이 아니라 shared bottleneck drop을 공통 혼잡 신호로 보고 S1/S2 agent에 반영합니다.
// -----------------------------------------------------------------------------

void QueueDropCallback(Ptr<GymTcpEnv> env, Ptr<const Packet> packet)
{
    env->ReportQueueDrop(1);
}


// -----------------------------------------------------------------------------
// Goodput logging
// 0.1초 control interval 동안 누적된 S1/S2 수신 byte를 Mbps로 변환해 goodput 로그 파일에 기록합니다.
// -----------------------------------------------------------------------------

void LogGoodput()
{
    double now = Simulator::Now().GetSeconds();

    double s1GoodputMbps =
        static_cast<double>(g_rxBytes[1]) * 8.0 / STEP_TIME / 1e6;

    double s2GoodputMbps =
        static_cast<double>(g_rxBytes[2]) * 8.0 / STEP_TIME / 1e6;

    if (f_goodput_s1.is_open())
    {
        f_goodput_s1 << now << " " << s1GoodputMbps << "\n";
    }

    if (f_goodput_s2.is_open())
    {
        f_goodput_s2 << now << " " << s2GoodputMbps << "\n";
    }

    g_rxBytes[1] = 0;
    g_rxBytes[2] = 0;
    g_rxBytes[3] = 0;

    if (!Simulator::IsFinished())
    {
        Simulator::Schedule(Seconds(STEP_TIME), &LogGoodput);
    }
}


// -----------------------------------------------------------------------------
// CWND trace logging
// 여기서는 S1/S2 TCP socket의 CongestionWindow 변화를 txt 파일로 저장합니다.
// -----------------------------------------------------------------------------

void PlotCwnd1(uint32_t oldVal, uint32_t newVal)
{
    if (f_cwnd_s1.is_open())
    {
        f_cwnd_s1 << Simulator::Now().GetSeconds() << " " << newVal << "\n";
    }
}

void PlotCwnd2(uint32_t oldVal, uint32_t newVal)
{
    if (f_cwnd_s2.is_open())
    {
        f_cwnd_s2 << Simulator::Now().GetSeconds() << " " << newVal << "\n";
    }
}


// -----------------------------------------------------------------------------
// RTT trace logging
// 여기서는 S1/S2 TCP socket의 RTT 변화를 ms 단위로 txt 파일에 저장합니다.
// -----------------------------------------------------------------------------

void PlotRtt1(Time oldVal, Time newVal)
{
    if (f_rtt_s1.is_open() && newVal.GetMilliSeconds() > 0)
    {
        f_rtt_s1 << Simulator::Now().GetSeconds()
                 << " " << newVal.GetMilliSeconds() << "\n";
    }
}

void PlotRtt2(Time oldVal, Time newVal)
{
    if (f_rtt_s2.is_open() && newVal.GetMilliSeconds() > 0)
    {
        f_rtt_s2 << Simulator::Now().GetSeconds()
                 << " " << newVal.GetMilliSeconds() << "\n";
    }
}


// -----------------------------------------------------------------------------
// TCP trace 연결
// 여기서는 NodeList index를 기준으로 S1/S2 TCP socket의 CWND와 RTT tracer를 연결합니다.
//
// NodeList ordering:
//   NodeList/0 = S1
//   NodeList/1 = S2
//   NodeList/2 = S3
//
// **For later uses - 수정이 필요할 때:
// topology node 생성 순서가 바뀌면 이 경로 검토하기
// -----------------------------------------------------------------------------

void ConnectTcpTracers()
{
    Config::ConnectWithoutContext(
        "/NodeList/0/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow",
        MakeCallback(&PlotCwnd1));

    Config::ConnectWithoutContext(
        "/NodeList/0/$ns3::TcpL4Protocol/SocketList/0/RTT",
        MakeCallback(&PlotRtt1));

    Config::ConnectWithoutContext(
        "/NodeList/1/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow",
        MakeCallback(&PlotCwnd2));

    Config::ConnectWithoutContext(
        "/NodeList/1/$ns3::TcpL4Protocol/SocketList/0/RTT",
        MakeCallback(&PlotRtt2));
}


// -----------------------------------------------------------------------------
// Bottleneck queue drop tracer 연결
// parking-lot 구조에서 핵심 shared bottleneck인 R2->R3 방향 queue의 drop event를 GymTcpEnv에 연결합니다.
// -----------------------------------------------------------------------------

void ConnectQueueDropTracer(Ptr<GymTcpEnv> env, const NetDeviceContainer &dR2R3)
{
    Ptr<PointToPointNetDevice> r2Device =
        DynamicCast<PointToPointNetDevice>(dR2R3.Get(0));

    if (!r2Device || !r2Device->GetQueue())
    {
        NS_LOG_WARN("Failed to connect R2->R3 queue drop tracer.");
        return;
    }

    r2Device->GetQueue()->TraceConnectWithoutContext(
        "Drop",
        MakeCallback(&QueueDropCallback).Bind(env));
}


// -----------------------------------------------------------------------------
// RL agent socket 등록 1회 시도
// 여기서는 특정 node의 TCP socket을 찾아 GymTcpEnv에 flowId/serviceType과 함께 등록합니다. 
// socket이 아직 생성되지 않은 경우 false를 반환합니다.
// -----------------------------------------------------------------------------

bool TryRegisterAgentOnce(
    Ptr<GymTcpEnv> env,
    uint32_t nodeId,
    uint32_t flowId,
    uint32_t serviceType)
{
    if (env->HasAgent(flowId))
    {
        return true;
    }

    std::ostringstream path;
    path << "/NodeList/" << nodeId
         << "/$ns3::TcpL4Protocol/SocketList/0";

    Config::MatchContainer match = Config::LookupMatches(path.str());

    if (match.GetN() == 0)
    {
        return false;
    }

    Ptr<TcpSocketBase> socket = DynamicCast<TcpSocketBase>(match.Get(0));

    if (!socket)
    {
        return false;
    }

    return env->AddTcpAgent(socket, flowId, serviceType);
}


// -----------------------------------------------------------------------------
// RL agent socket 등록 재시도
// application start 직후 TCP socket이 아직 생성되지 않았을 수 있으므로,
// 일정 시간까지 0.2초 간격으로 socket 등록을 재시도합니다.
// -----------------------------------------------------------------------------

void TryRegisterAgentWithRetry(
    Ptr<GymTcpEnv> env,
    uint32_t nodeId,
    uint32_t flowId,
    uint32_t serviceType,
    double deadline)
{
    bool success = TryRegisterAgentOnce(env, nodeId, flowId, serviceType);

    if (success)
    {
        return;
    }

    if (Simulator::Now().GetSeconds() < deadline)
    {
        Simulator::Schedule(
            Seconds(0.2),
            &TryRegisterAgentWithRetry,
            env,
            nodeId,
            flowId,
            serviceType,
            deadline);
    }
    else
    {
        NS_LOG_WARN("Failed to register RL agent before deadline. flowId=" << flowId);
    }
}

} // namespace


// -----------------------------------------------------------------------------
// Main simulation entry point
// mode 인자를 받아 Baseline 또는 RL simulation을 실행합니다.
//
// Baseline mode:
//   - TCP Cubic만 사용
//   - Gym interface 연결 없음
//   - baseline_*.txt 로그 생성
//
// RL mode:
//   - TCP Cubic 기반 socket에서 시작
//   - S1/S2 TCP socket을 GymTcpEnv에 등록
//   - Python PPO agent가 S1/S2 CWND를 보조 제어
//   - rl_*.txt 로그 생성
// -----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    std::string mode = "RL";

    CommandLine cmd;
    cmd.AddValue("mode", "Baseline or RL", mode);
    cmd.Parse(argc, argv);

    Config::SetDefault(
        "ns3::TcpL4Protocol::SocketType",
        StringValue("ns3::TcpCubic"));

    std::string prefix = (mode == "RL") ? "rl" : "baseline";
    OpenResultFiles(prefix);


    // -------------------------------------------------------------------------
    // Node 생성
    // 여기서는 source node 3개, router node 3개, destination node 3개를 생성합니다.
    //
    // S0 = S1 FTP/background
    // S1 = S2 primary video-like
    // S2 = S3 burst cross traffic
    // -------------------------------------------------------------------------

    NodeContainer S;
    NodeContainer R;
    NodeContainer D;

    S.Create(3);
    R.Create(3);
    D.Create(3);


    // -------------------------------------------------------------------------
    // Link helper 설정
    // access link는 충분히 넓게 두고, bottleneck link는 15Mbps/10ms/150p로 설정했습니다.
    // -------------------------------------------------------------------------

    PointToPointHelper access;
    PointToPointHelper bottleneck;

    access.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    access.SetChannelAttribute("Delay", StringValue("2ms"));

    bottleneck.SetDeviceAttribute("DataRate", StringValue(BOTTLENECK_RATE));
    bottleneck.SetChannelAttribute("Delay", StringValue("10ms"));
    bottleneck.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue(QUEUE_SIZE));


    // -------------------------------------------------------------------------
    // Parking-lot topology 구성 [Diagram**]
    //
    // S1 ---- R1 ==== R2 ==== R3 ---- D1
    //                /  \       \
    //              S2    S3      D2/D3
    //
    // S1: FTP/background, RL-controlled
    // S2: primary video-like flow, RL-controlled and protected
    // S3: burst cross traffic, not RL-controlled
    //
    // R2->R3는 S1/S2/S3가 함께 경쟁하는 shared bottleneck으로 구성했습니다.
    // -------------------------------------------------------------------------

    NetDeviceContainer dS1R1 = access.Install(S.Get(0), R.Get(0));
    NetDeviceContainer dR1R2 = bottleneck.Install(R.Get(0), R.Get(1));
    NetDeviceContainer dS2R2 = access.Install(S.Get(1), R.Get(1));
    NetDeviceContainer dS3R2 = access.Install(S.Get(2), R.Get(1));
    NetDeviceContainer dR2R3 = bottleneck.Install(R.Get(1), R.Get(2));
    NetDeviceContainer dR3D1 = access.Install(R.Get(2), D.Get(0));
    NetDeviceContainer dR3D2 = access.Install(R.Get(2), D.Get(1));
    NetDeviceContainer dR3D3 = access.Install(R.Get(2), D.Get(2));


    // -------------------------------------------------------------------------
    // Internet stack 및 IP 주소 할당
    // 모든 node에 TCP/IP stack을 설치하고 각 point-to-point link에 서로 다른 subnet을 할당합니다.
    // -------------------------------------------------------------------------

    InternetStackHelper internet;
    internet.InstallAll();

    Ipv4AddressHelper address;

    address.SetBase("10.1.1.0", "255.255.255.0");
    address.Assign(dS1R1);

    address.SetBase("10.1.2.0", "255.255.255.0");
    address.Assign(dR1R2);

    address.SetBase("10.1.3.0", "255.255.255.0");
    address.Assign(dS2R2);

    address.SetBase("10.1.4.0", "255.255.255.0");
    address.Assign(dS3R2);

    address.SetBase("10.1.5.0", "255.255.255.0");
    address.Assign(dR2R3);

    address.SetBase("10.1.6.0", "255.255.255.0");
    Ipv4InterfaceContainer iR3D1 = address.Assign(dR3D1);

    address.SetBase("10.1.7.0", "255.255.255.0");
    Ipv4InterfaceContainer iR3D2 = address.Assign(dR3D2);

    address.SetBase("10.1.8.0", "255.255.255.0");
    Ipv4InterfaceContainer iR3D3 = address.Assign(dR3D3);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();


    // -------------------------------------------------------------------------
    // GymTcpEnv 및 sink application 구성
    // 여기서는 GymTcpEnv를 만들고, D1/D2/D3에 TCP PacketSink를 설치합니다.
    // 각 sink의 Rx callback은 flowId별 수신 byte를 기록합니다.
    // -------------------------------------------------------------------------

    Ptr<GymTcpEnv> env = CreateObject<GymTcpEnv>();

    PacketSinkHelper sink(
        "ns3::TcpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), 9));

    for (uint32_t i = 0; i < 3; ++i)
    {
        ApplicationContainer sinkApp = sink.Install(D.Get(i));
        sinkApp.Start(Seconds(0.0));

        sinkApp.Get(0)->TraceConnectWithoutContext(
            "Rx",
            MakeCallback(&RxCallback).Bind(env, i + 1));
    }


    // -------------------------------------------------------------------------
    // S1 FTP/background traffic 생성
    // S1은 항상 전송되는 best-effort background flow입니다.
    // RL mode에서는 S1도 제어 대상이지만, starvation은 방지되도록 했습니다.
    // -------------------------------------------------------------------------

    BulkSendHelper src1(
        "ns3::TcpSocketFactory",
        InetSocketAddress(iR3D1.GetAddress(1), 9));

    src1.SetAttribute("SendSize", UintegerValue(PACKET_SIZE));

    ApplicationContainer s1App = src1.Install(S.Get(0));
    s1App.Start(Seconds(0.0));
    s1App.Stop(Seconds(SIM_TIME));


    // -------------------------------------------------------------------------
    // S2 primary video-like traffic 생성
    // S2는 보호 대상 flow입니다. 
    // 핵심 QoS 목표는 S2가 burst 구간에서도 5Mbps goodput과 120ms RTT constraint를 최대한 만족하도록 하는 것입니다.
    // -------------------------------------------------------------------------

    BulkSendHelper src2(
        "ns3::TcpSocketFactory",
        InetSocketAddress(iR3D2.GetAddress(1), 9));

    src2.SetAttribute("SendSize", UintegerValue(PACKET_SIZE));

    ApplicationContainer s2App = src2.Install(S.Get(1));
    s2App.Start(Seconds(0.0));
    s2App.Stop(Seconds(SIM_TIME));


    // -------------------------------------------------------------------------
    // S3 burst cross traffic 생성
    // S3는 50~100초에만 유입되는 burst traffic이며 RL 제어 대상이 아닙니다.
    // 이 flow는 S2 primary video-like flow에 혼잡을 유발하는 외부 traffic으로 사용됩니다.
    // -------------------------------------------------------------------------

    BulkSendHelper src3(
        "ns3::TcpSocketFactory",
        InetSocketAddress(iR3D3.GetAddress(1), 9));

    src3.SetAttribute("SendSize", UintegerValue(PACKET_SIZE));

    ApplicationContainer s3App = src3.Install(S.Get(2));
    s3App.Start(Seconds(50.0));
    s3App.Stop(Seconds(100.0));


    // -------------------------------------------------------------------------
    // Logging 및 queue drop tracer 설정
    // goodput logging, CWND/RTT trace, bottleneck queue drop trace를 등록합니다.
    // -------------------------------------------------------------------------

    ConnectQueueDropTracer(env, dR2R3);

    Simulator::Schedule(Seconds(STEP_TIME), &LogGoodput);
    Simulator::Schedule(Seconds(1.0), &ConnectTcpTracers);


    // -------------------------------------------------------------------------
    // RL mode 설정
    // RL mode에서는 S1/S2 TCP socket을 GymTcpEnv에 등록하고, OpenGymInterface를 통해 Python PPO agent와 연결합니다.
    // S3 burst traffic은 RL 제어 대상이 아니므로 agent로 등록하지 않습니다.
    // -------------------------------------------------------------------------

    if (mode == "RL")
    {
        Simulator::Schedule(
            Seconds(0.2),
            &TryRegisterAgentWithRetry,
            env,
            0,
            1,
            SERVICE_FTP,
            5.0);

        Simulator::Schedule(
            Seconds(0.2),
            &TryRegisterAgentWithRetry,
            env,
            1,
            2,
            SERVICE_VIDEO,
            5.0);

        Ptr<OpenGymInterface> openGym = CreateObject<OpenGymInterface>(5555);
        env->SetOpenGymInterface(openGym);

        Simulator::Schedule(Seconds(STEP_TIME), &GymStep, env);
    }


    // -------------------------------------------------------------------------
    // Simulation 실행 및 종료 처리
    // 여기서는 설정 정보를 출력하고, 160초까지 simulation을 실행한 뒤 로그 파일을 닫습니다.
    // -------------------------------------------------------------------------

    NS_LOG_UNCOND("Mode: " << mode
                  << ", TCP: TcpCubic"
                  << ", Bottleneck: " << BOTTLENECK_RATE
                  << ", Queue: " << QUEUE_SIZE
                  << ", Step: " << STEP_TIME
                  << "s, SimTime: " << SIM_TIME << "s");

    Simulator::Stop(Seconds(SIM_TIME));
    Simulator::Run();
    Simulator::Destroy();

    CloseResultFiles();

    return 0;
}