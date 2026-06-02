#ifndef GYM_TCP_ENV_H
#define GYM_TCP_ENV_H

#include "ns3/core-module.h"
#include "ns3/opengym-module.h"
#include "ns3/tcp-socket-base.h"

#include <vector>

using namespace ns3;


// -----------------------------------------------------------------------------
// Service type 정의
// flow의 서비스 성격을 구분지어둡니다. 
// reward wrapper와 결과 해석에서는 S1을 FTP/background, S2를 video-like flow로 다루기에 serviceType 값을 고정해 사용합니다.
// -----------------------------------------------------------------------------

enum ServiceType
{
    SERVICE_FTP = 1,
    SERVICE_VIDEO = 2
};


// -----------------------------------------------------------------------------
// TCP agent 상태 저장 structure
// 여기서는 RL 제어 대상이 되는 각 TCP flow의 socket과 현재 상태를 저장합니다.
// topology 파일에서 S1/S2 TCP socket을 AddTcpAgent()로 등록하면, GymTcpEnv는 이 구조체를 통해 CWND, RTT, 수신 byte, loss count를 관리합니다.
// -----------------------------------------------------------------------------

struct TcpAgentData
{
    Ptr<TcpSocketBase> socket;  // ns-3 TCP socket pointer
    uint32_t flowId;           // 1: S1 FTP/background, 2: S2 primary video-like
    uint32_t serviceType;      // 1: FTP, 2: Video

    uint32_t currentCwnd;      // current congestion window
    double currentRtt;         // current RTT in ms
    Time minRtt;               // minimum observed RTT, used for RTT ratio

    uint32_t bytesReceived;    // bytes received during the current control interval
    uint32_t segmentLossCount; // queue drop / segment loss signal for this interval
};


// -----------------------------------------------------------------------------
// GymTcpEnv 클래스 선언
// 여기서는 ns-3 TCP 상태를 ns3-gym observation/action interface로 연결합니다.
// 역할:
// - topology 파일에서 등록한 S1/S2 TCP socket을 RL agent로 관리
// - cwnd, RTT, RTT ratio, bytes, loss, serviceType을 observation으로 구성
// - Python PPO agent가 선택한 discrete action을 TCP CWND 조정으로 적용
// - ReportRx(), ReportQueueDrop()을 통해 goodput/loss 관련 상태 갱신
// 실제 reward shaping과 action projection은 Python의 qos_reward_env.py에서 수행되고, 이 클래스는 ns-3 쪽 observation/action 징검다리 역할을 담당합니다.
// -----------------------------------------------------------------------------

class GymTcpEnv : public OpenGymEnv
{
public:
    GymTcpEnv();
    virtual ~GymTcpEnv();


    // -------------------------------------------------------------------------
    // TCP agent 등록 및 조회
    // topology 파일에서 생성된 TCP socket을 flowId/serviceType과 함께 등록합니다.
    // 현재 실험에서는 S1과 S2만을 RL 제어 대상으로 등록합니다.
    // -------------------------------------------------------------------------

    bool AddTcpAgent(
        Ptr<TcpSocketBase> socket,
        uint32_t flowId,
        uint32_t serviceType);

    bool HasAgent(uint32_t flowId) const;


    // -------------------------------------------------------------------------
    // 수신 byte 및 queue drop 보고
    // PacketSink Rx callback은 ReportRx()를 호출해 control interval 동안의 수신 byte를 누적하고, 
    // Queue drop callback은 ReportQueueDrop()을 호출해 loss signal을 갱신하는 용도입니다.
    // -------------------------------------------------------------------------

    void ReportRx(uint32_t flowId, uint32_t bytes);
    void ReportQueueDrop(uint32_t count = 1);


    // -------------------------------------------------------------------------
    // ns3-gym interface override
    // Python PPO agent와 상호작용하기 위해 OpenGymEnv에서 요구하는 함수들입니다.
    // GetObservation()은 현재 TCP 상태를 Python으로 넘기고,
    // ExecuteActions()는 Python에서 받은 action을 ns-3 TCP socket에 적용합니다.
    // -------------------------------------------------------------------------

    Ptr<OpenGymSpace> GetObservationSpace() override;
    Ptr<OpenGymSpace> GetActionSpace() override;
    Ptr<OpenGymDataContainer> GetObservation() override;
    bool ExecuteActions(Ptr<OpenGymDataContainer> action) override;
    float GetReward() override;
    bool GetGameOver() override;
    std::string GetExtraInfo() override;


private:

    // -------------------------------------------------------------------------
    // TCP trace callback
    // 등록된 TCP socket의 CongestionWindow와 RTT 변화를 추적해 TcpAgentData의 currentCwnd/currentRtt 값을 갱신합니다.
    // -------------------------------------------------------------------------

    void CwndTracer(uint32_t flowId, uint32_t oldVal, uint32_t newVal);
    void RttTracer(uint32_t flowId, Time oldVal, Time newVal);


    // -------------------------------------------------------------------------
    // Observation 구성
    // 특정 flow의 상태를 OpenGymBoxContainer에 추가합니다.
    // flow별 observation layout:
    //   [0] cwnd
    //   [1] rtt
    //   [2] rttRatio
    //   [3] bytesReceived
    //   [4] segmentLossCount
    //   [5] serviceType
    // S1과 S2가 순서대로 들어가므로 Python wrapper에서는 총 12차원으로 읽습니다.
    // -------------------------------------------------------------------------

    void AddAgentObservation(
        Ptr<OpenGymBoxContainer<float>> box,
        uint32_t flowId,
        uint32_t serviceType);


    // -------------------------------------------------------------------------
    // Action 적용
    // Python PPO agent가 선택한 action을 특정 flow의 CWND 조정으로 적용합니다.
    // agentAction 의미:
    //   0 = decrease CWND
    //   1 = hold CWND
    //   2 = increase CWND
    // action projection은 Python wrapper에서 먼저 수행되며, 이 함수는 projection 이후의 최종 action을 socket에 반영합니다.
    // -------------------------------------------------------------------------

    void ApplyAgentAction(uint32_t flowId, uint32_t agentAction);


    // -------------------------------------------------------------------------
    // 환경 내부 상태
    // m_agents는 현재 등록된 TCP flow들의 상태를 저장합니다.
    // m_simulationTime은 GetGameOver() 등에서 episode 종료 판단에 사용됩니다.
    // -------------------------------------------------------------------------

    std::vector<TcpAgentData> m_agents;
    double m_simulationTime;
};

#endif
