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

NS_LOG_COMPONENT_DEFINE("Topo3StarGateway");

namespace
{

// -----------------------------------------------------------------------------
// 실험 공통 상수 및 결과 저장 경로
// control interval, simulation time, packet size, bottleneck을 설정하고 결과 파일 저장 위치를 정의합니다.
// 이 topology는 학습에 포함하지 않은 unseen evaluation topology로 사용합니다.
// -----------------------------------------------------------------------------

constexpr double STEP_TIME = 0.1;
constexpr double SIM_TIME = 160.0;
constexpr uint32_t PACKET_SIZE = 536;

const std::string RESULT_DIR = "scratch/results/star_gateway_rtt/";
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
//   3 = S3 other-user burst cross traffic
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
// 0.1초마다 GymTcpEnv::Notify()를 호출해 Python agent와 observation/action interaction을 수행합니다.
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
// S3는 GymTcpEnv에 agent로 등록하지 않으므로 RL 제어 대상에는 포함되지 않습니다.
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
// shared uplink queue에서 drop이 발생하면 GymTcpEnv에 loss signal을 전달합니다.
// 현재 구현은 flow별 loss attribution이 아니라 shared bottleneck drop을 공통 혼잡 신호로 보고 S1/S2 agent에 반영합니다.
// -----------------------------------------------------------------------------

void QueueDropCallback(Ptr<GymTcpEnv> env, Ptr<const Packet> packet)
{
    env->ReportQueueDrop(1);
}


// -----------------------------------------------------------------------------
// Goodput logging
// 0.1초 control interval 동안 누적된 S1/S2 수신 byte를 Mbps로 변환해 goodput 로그 파일에 기록합니다.
// S3는 외부 burst traffic으로 사용되며, 최종 QoS 분석 대상은 S1/S2입니다.
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
// S1/S2 TCP socket의 CongestionWindow 변화를 txt 파일로 저장합니다.
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
// S1/S2 TCP socket의 RTT 변화를 ms 단위로 txt 파일에 저장합니다.
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
// NodeList index를 기준으로 S1/S2 TCP socket의 CWND와 RTT tracer를 연결합니다.
// NodeList ordering:
//   NodeList/0 = S1
//   NodeList/1 = S2
//   NodeList/2 = S3
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
// Shared uplink queue drop tracer 연결
// star-like gateway 구조에서 bottleneck인 Gateway->ISP/Core 방향 queue의 drop event를 GymTcpEnv에 연결합니다.
// Traffic direction: S1/S2/S3 -> AP -> Gateway -> ISP/Core -> D1/D2/D3
// 그렇기에 congested output queue는 dGwIsp.Get(0), 다시 말해 Gateway -> ISP/Core 방향입니다.
// -----------------------------------------------------------------------------

void ConnectQueueDropTracer(Ptr<GymTcpEnv> env, const NetDeviceContainer &dGwIsp)
{
    Ptr<PointToPointNetDevice> gwDevice =
        DynamicCast<PointToPointNetDevice>(dGwIsp.Get(0));

    if (!gwDevice || !gwDevice->GetQueue())
    {
        NS_LOG_WARN("Failed to connect Gateway->ISP queue drop tracer.");
        return;
    }

    gwDevice->GetQueue()->TraceConnectWithoutContext(
        "Drop",
        MakeCallback(&QueueDropCallback).Bind(env));
}


// -----------------------------------------------------------------------------
// RL agent socket 등록 1회 시도
// 특정 node의 TCP socket을 찾아 GymTcpEnv에 flowId/serviceType과 함께 등록합니다. 
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
// application start 직후 TCP socket이 아직 생성되지 않았을 수 있으니 일정 시간까지 0.2초 간격으로 socket 등록을 재시도합니다.
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
// mode 정보를 받아 Baseline 또는 RL simulation을 실행합니다.
// Baseline mode:
//   - TCP Cubic만 사용
//   - Gym interface 연결 없음
//   - baseline_*.txt 로그 생성
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
    // source node 3개, gateway 계층 node 3개, destination node 3개를 생성합니다.
    // S.Get(0) = S1 FTP/background
    // S.Get(1) = S2 primary video-like
    // S.Get(2) = S3 other-user burst traffic
    // G.Get(0) = AP
    // G.Get(1) = Gateway
    // G.Get(2) = ISP/Core
    // -------------------------------------------------------------------------

    NodeContainer S;
    NodeContainer G;
    NodeContainer D;

    S.Create(3);
    G.Create(3);
    D.Create(3);

    Ptr<Node> ap = G.Get(0);
    Ptr<Node> gateway = G.Get(1);
    Ptr<Node> isp = G.Get(2);


    // -------------------------------------------------------------------------
    // Link helper 설정
    // 사용자별 local access delay를 다르게 줘서 shared gateway 환경의 heterogeneous device/user 특성을 단순화합니다.
    // 실제 Wi-Fi PHY/MAC을 모델링하지는 못했고, 여러 사용자가 AP/Gateway와 ISP uplink를 공유하는 상황을 wired point-to-point link로 단순화해서 나타내본 abstraction입니다.
    // -------------------------------------------------------------------------

    PointToPointHelper accessS1;
    PointToPointHelper accessS2;
    PointToPointHelper accessS3;
    PointToPointHelper lan;
    PointToPointHelper bottleneck;
    PointToPointHelper serverAccess;

    accessS1.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    accessS1.SetChannelAttribute("Delay", StringValue("2ms"));

    accessS2.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    accessS2.SetChannelAttribute("Delay", StringValue("1ms"));

    accessS3.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    accessS3.SetChannelAttribute("Delay", StringValue("4ms"));

    lan.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    lan.SetChannelAttribute("Delay", StringValue("1ms"));

    bottleneck.SetDeviceAttribute("DataRate", StringValue(BOTTLENECK_RATE));
    bottleneck.SetChannelAttribute("Delay", StringValue("10ms"));
    bottleneck.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue(QUEUE_SIZE));

    serverAccess.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    serverAccess.SetChannelAttribute("Delay", StringValue("2ms"));


    // -------------------------------------------------------------------------
    // Star-like shared gateway topology 구성 [참고용 figure***]
    //
    // S1 FTP/background ─┐
    // S2 primary video  ─┼── AP -- Gateway ===== ISP/Core ─┬── D1
    // S3 burst traffic  ─┘                                 ├── D2
    //                                                       └── D3
    //
    // Gateway->ISP/Core는 여러 사용자가 공유하는 uplink bottleneck입니다.
    // -------------------------------------------------------------------------

    NetDeviceContainer dS1Ap = accessS1.Install(S.Get(0), ap);
    NetDeviceContainer dS2Ap = accessS2.Install(S.Get(1), ap);
    NetDeviceContainer dS3Ap = accessS3.Install(S.Get(2), ap);
    NetDeviceContainer dApGw = lan.Install(ap, gateway);
    NetDeviceContainer dGwIsp = bottleneck.Install(gateway, isp);
    NetDeviceContainer dIspD1 = serverAccess.Install(isp, D.Get(0));
    NetDeviceContainer dIspD2 = serverAccess.Install(isp, D.Get(1));
    NetDeviceContainer dIspD3 = serverAccess.Install(isp, D.Get(2));


    // -------------------------------------------------------------------------
    // Internet stack 및 IP 주소 할당
    // 모든 node에 TCP/IP stack을 설치하고 각 point-to-point link에 서로 다른 subnet을 할당합니다.
    // -------------------------------------------------------------------------

    InternetStackHelper internet;
    internet.InstallAll();

    Ipv4AddressHelper address;

    address.SetBase("10.3.1.0", "255.255.255.0");
    address.Assign(dS1Ap);

    address.SetBase("10.3.2.0", "255.255.255.0");
    address.Assign(dS2Ap);

    address.SetBase("10.3.3.0", "255.255.255.0");
    address.Assign(dS3Ap);

    address.SetBase("10.3.4.0", "255.255.255.0");
    address.Assign(dApGw);

    address.SetBase("10.3.5.0", "255.255.255.0");
    address.Assign(dGwIsp);

    address.SetBase("10.3.6.0", "255.255.255.0");
    Ipv4InterfaceContainer iIspD1 = address.Assign(dIspD1);

    address.SetBase("10.3.7.0", "255.255.255.0");
    Ipv4InterfaceContainer iIspD2 = address.Assign(dIspD2);

    address.SetBase("10.3.8.0", "255.255.255.0");
    Ipv4InterfaceContainer iIspD3 = address.Assign(dIspD3);

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();


    // -------------------------------------------------------------------------
    // GymTcpEnv 및 sink application 구성
    // GymTcpEnv를 만들고, D1/D2/D3에 TCP PacketSink를 설치합니다.
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
    // S1은 항상 전송되는 best-effort 또는 보조 traffic입니다.
    // RL mode에서는 S1도 제어 대상이지만, 완전히 끊기지는 않도록 조정하게 됩니다.
    // -------------------------------------------------------------------------

    BulkSendHelper src1(
        "ns3::TcpSocketFactory",
        InetSocketAddress(iIspD1.GetAddress(1), 9));

    src1.SetAttribute("SendSize", UintegerValue(PACKET_SIZE));

    ApplicationContainer s1App = src1.Install(S.Get(0));
    s1App.Start(Seconds(0.0));
    s1App.Stop(Seconds(SIM_TIME));


    // -------------------------------------------------------------------------
    // S2 primary video-like traffic 생성 (S2 보호가 목적)
    // -------------------------------------------------------------------------

    BulkSendHelper src2(
        "ns3::TcpSocketFactory",
        InetSocketAddress(iIspD2.GetAddress(1), 9));

    src2.SetAttribute("SendSize", UintegerValue(PACKET_SIZE));

    ApplicationContainer s2App = src2.Install(S.Get(1));
    s2App.Start(Seconds(0.0));
    s2App.Stop(Seconds(SIM_TIME));


    // -------------------------------------------------------------------------
    // S3 other-user burst traffic 생성
    // S3는 50~100초에만 유입되는 다른 사용자 burst traffic이고 RL 제어 대상으로 두지 않습니다
    // -------------------------------------------------------------------------

    BulkSendHelper src3(
        "ns3::TcpSocketFactory",
        InetSocketAddress(iIspD3.GetAddress(1), 9));

    src3.SetAttribute("SendSize", UintegerValue(PACKET_SIZE));

    ApplicationContainer s3App = src3.Install(S.Get(2));
    s3App.Start(Seconds(50.0));
    s3App.Stop(Seconds(100.0));


    // -------------------------------------------------------------------------
    // Logging 및 queue drop tracer 설정
    // goodput logging, CWND/RTT trace, shared uplink queue drop trace를 등록합니다.
    // -------------------------------------------------------------------------

    ConnectQueueDropTracer(env, dGwIsp);

    Simulator::Schedule(Seconds(STEP_TIME), &LogGoodput);
    Simulator::Schedule(Seconds(1.0), &ConnectTcpTracers);


    // -------------------------------------------------------------------------
    // RL mode 설정
    // RL mode에서는 S1/S2 TCP socket을 GymTcpEnv에 등록하고, OpenGymInterface를 통해 Python PPO agent와 연결합니다.
    // S3 burst traffic은 외부 traffic으로 보고, 제어하는 요소가 아니기에 agent로 등록하지 않습니다.
    // 최종 결과 출력에는 eval-only를 사용했기에 사용되지 않은 부분입니다.
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
    // 설정 정보를 출력하고, 160초까지 simulation을 실행한 뒤 로그 파일을 닫습니다.
    // -------------------------------------------------------------------------

    NS_LOG_UNCOND("Mode: " << mode
                  << ", Topology: Star-like Shared Gateway"
                  << ", TCP: TcpCubic"
                  << ", Bottleneck: " << BOTTLENECK_RATE
                  << ", Queue: " << QUEUE_SIZE
                  << ", Step: " << STEP_TIME
                  << "s, SimTime: " << SIM_TIME << "s"
                  << ", ResultDir: " << RESULT_DIR);

    Simulator::Stop(Seconds(SIM_TIME));
    Simulator::Run();
    Simulator::Destroy();

    CloseResultFiles();

    return 0;
}
