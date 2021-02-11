#include <cmath>
#include <algorithm>

#if defined(_MSC_VER)
	#include <intrin.h>
#else
	#include <xmmintrin.h>
	#include <emmintrin.h>
	#include <immintrin.h>
#endif

/* インクルードガード */
#ifndef	GLSLVMATH_H
#define	GLSLVMATH_H


namespace glslmath {


template<typename Element0_t, typename Traits0_t, int recursiveCount0>
union Vec;

template<typename Element0_t, int opeDimC0, int memDimC0, int opeDimR0, int memDimR0, int recursiveCount0>
union Mat;

template<typename Element0_t, typename Traits0_t, int recursiveCount0>
union Ivec;

template<typename Element0_t, typename Traits0_t, int recursiveCount0>
union Bvec;


/*=============================================================================
▼	型情報を保持する構造体
-----------------------------------------------------------------------------*/
template<int opeDim_, int memDim_, int swizzle_>
struct Traits {
	enum {
		/* 演算次元数 */
		opeDim = opeDim_,

		/* メモリ次元数 */
		memDim = memDim_,

		/* SWIZZLE */
		swizzle = swizzle_,

		/* 各メンバのインデクス */
		i0 =  swizzle        & 15,
		i1 = (swizzle >>  4) & 15,
		i2 = (swizzle >>  8) & 15,
		i3 = (swizzle >> 12) & 15,

		/* メンバ重複があるか？ */
		hasDuplicatedMember = (
			(opeDim >= 4 && (i3 == i0 || i3 == i1 || i3 == i2))
		||	(opeDim >= 3 && (i2 == i0 || i2 == i1))
		||	(opeDim >= 2 && (i1 == i0))
		),

		/* メモリ次元数を超えるインデクスが存在しないなら妥当 */
		isValid	= !(
			(opeDim >= 1 && i0 >= memDim)
		||	(opeDim >= 2 && i1 >= memDim)
		||	(opeDim >= 3 && i2 >= memDim)
		||	(opeDim >= 4 && i3 >= memDim)
		)
	};
};


/*=============================================================================
▼	指定次元数のベクトル型を生成
-----------------------------------------------------------------------------*/
template<typename Element_t, int dim>
using GenVec = Vec<Element_t, Traits<dim, dim, 0x03210>, 0>;

template<typename Element_t, int opeDim, int memDim>
using GenVec2 = Vec<Element_t, Traits<opeDim, memDim, 0x03210>, 0>;

template<typename Element_t, int dim>
using GenIvec = Ivec<Element_t, Traits<dim, dim, 0x03210>, 0>;

template<typename Element_t, int dim>
using GenBvec = Bvec<Element_t, Traits<dim, dim, 0x03210>, 0>;


/*=============================================================================
▼	指定次元数の行列型を生成
-----------------------------------------------------------------------------*/
template<typename Element_t, int opeDimC, int opeDimR>
using GenMat = Mat<Element_t, opeDimC, opeDimC, opeDimR, opeDimR, 0>;


/*=============================================================================
▼	メンバアクセス用
-----------------------------------------------------------------------------*/
template<typename Element0_t, int memDim0, int index0>
class ScalarInVec {
private:
	/* 自分自身の型 */
	using This_t = ScalarInVec<Element0_t, memDim0, index0>;

public:
	/* ベクトル成分 */
	Element0_t elements[memDim0];

	/* スカラ出力 */
	inline operator Element0_t(){
		static_assert(index0 < memDim0);
		return this->elements[index0];
	}
	inline operator Element0_t() const {
		static_assert(index0 < memDim0);
		return this->elements[index0];
	}

	/* スカラから代入 */
	inline This_t& operator=(
		Element0_t param
	){
		static_assert(index0 < memDim0);
		this->elements[index0] = param;
		return *this;
	}

	/* ScalarInVec から代入 */
	template<int memDim1, int index1>
	inline This_t& operator=(
		const ScalarInVec<Element0_t, memDim1, index1> &rParam
	){
		static_assert(index0 < memDim0);
		static_assert(index1 < memDim1);
		this->elements[index0] = rParam.elements[index1];
		return *this;
	}

	/* キャスト（暗黙に行われるスカラからのキャスト）*/
	ScalarInVec<Element0_t, memDim0, index0>(
		Element0_t param
	){
		static_assert(index0 < memDim0);
		this->elements[index0] = param;
	}
};

/* address-of 演算子 */
template<typename Element0_t, int memDim0, int index0>
static inline Element0_t * operator&(
	ScalarInVec<Element0_t, memDim0, index0> &rThis
){
	static_assert(index0 < memDim0);
	return &rThis.elements[index0];
}

/* address-of 演算子（const 版）*/
template<typename Element0_t, int memDim0, int index0>
static inline const Element0_t * operator&(
	const ScalarInVec<Element0_t, memDim0, index0> &rThis
){
	static_assert(index0 < memDim0);
	return &rThis.elements[index0];
}


/*=============================================================================
▼	SIMD 型
-----------------------------------------------------------------------------*/
template<typename Element_t, int dim>
struct GenSimdVecTraits {
	typedef Element_t IntrinsicType_t[dim];
	enum {
		isM128 = 0,
		isM256 = 0,
	};
};

template<>
struct GenSimdVecTraits<float, 4> {
	typedef __m128 IntrinsicType_t;
	enum {
		isM128 = 1,
		isM256 = 0,
	};
};

template<>
struct GenSimdVecTraits<double, 4> {
	typedef __m256 IntrinsicType_t;
	enum {
		isM128 = 0,
		isM256 = 1,
	};
};


/*=============================================================================
▼	ベクトルクラスコード共通部分
-----------------------------------------------------------------------------*/
/* 冒頭共通部分 */
#define IMPL_VEC_COMMON(Type_t)\
	private:\
		/* 自分自身の型 */\
		using This_t = Type_t<Element0_t, Traits0_t, recursiveCount0>;\
\
		/* 中間値の型（SWIZZLE 解決済み）*/\
		using Temp_t = Gen##Type_t<Element0_t, Traits0_t::opeDim>;\
\
		/* SIMD 型 */\
		using SimdVecTraits_t = GenSimdVecTraits<Element0_t, Traits0_t::memDim>;\
\
	public:\
		/* ベクトル成分 */\
		Element0_t elements[Traits0_t::memDim];\
\
		/* SIMD 型 */\
		typename SimdVecTraits_t::IntrinsicType_t simdVec;\
\
		/* this ポインタを取得 */\
		inline       This_t * GetThisPointer()       { return this; }\
		inline const This_t * GetThisPointer() const { return this; }\

/* [] アクセス */
#define IMPL_VEC_ARRAY_SUBSCRIPT_OPERATOR()\
	/* 非 const アクセス */\
	inline Element0_t& operator[](unsigned int iElement){\
		/* SWIZZLE 指定後の [] アクセスは illegal */\
		static_assert(recursiveCount0 == 0);\
\
		/* SWIZZLE を無視したアクセス */\
		return this->elements[iElement];\
	}\
\
	/* const アクセス */\
	inline const Element0_t& operator[](unsigned int iElement) const {\
		/* SWIZZLE 指定後の [] アクセスは illegal */\
		static_assert(recursiveCount0 == 0);\
\
		/* SWIZZLE を無視したアクセス */\
		return this->elements[iElement];\
	}\

/* 単項演算 */
#define IMPL_VEC_UNARY_OPERATOR(OP)\
	inline Temp_t operator OP() const {\
		Temp_t result;\
		static_assert(Traits0_t::isValid);\
		if constexpr (Traits0_t::opeDim >= 1) { result.elements[0] = OP this->elements[Traits0_t::i0]; }\
		if constexpr (Traits0_t::opeDim >= 2) { result.elements[1] = OP this->elements[Traits0_t::i1]; }\
		if constexpr (Traits0_t::opeDim >= 3) { result.elements[2] = OP this->elements[Traits0_t::i2]; }\
		if constexpr (Traits0_t::opeDim >= 4) { result.elements[3] = OP this->elements[Traits0_t::i3]; }\
		return result;\
	}\

/* 二項演算 */
#define IMPL_VEC_BINARY_OPERATOR(OP)\
	inline Temp_t operator OP(\
		const Temp_t &rParam\
	) const {\
		Temp_t result;\
		static_assert(Traits0_t::isValid);\
		if constexpr (Traits0_t::opeDim >= 1) { result.elements[0] = this->elements[Traits0_t::i0] OP rParam.elements[0]; }\
		if constexpr (Traits0_t::opeDim >= 2) { result.elements[1] = this->elements[Traits0_t::i1] OP rParam.elements[1]; }\
		if constexpr (Traits0_t::opeDim >= 3) { result.elements[2] = this->elements[Traits0_t::i2] OP rParam.elements[2]; }\
		if constexpr (Traits0_t::opeDim >= 4) { result.elements[3] = this->elements[Traits0_t::i3] OP rParam.elements[3]; }\
		return result;\
	}\

/* 二項演算（スカラ）*/
#define IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR(OP)\
	inline Temp_t operator OP(\
		const Element0_t param\
	) const {\
		Temp_t result;\
		static_assert(Traits0_t::isValid);\
		if constexpr (Traits0_t::opeDim >= 1) { result.elements[0] = this->elements[Traits0_t::i0] OP param; }\
		if constexpr (Traits0_t::opeDim >= 2) { result.elements[1] = this->elements[Traits0_t::i1] OP param; }\
		if constexpr (Traits0_t::opeDim >= 3) { result.elements[2] = this->elements[Traits0_t::i2] OP param; }\
		if constexpr (Traits0_t::opeDim >= 4) { result.elements[3] = this->elements[Traits0_t::i3] OP param; }\
		return result;\
	}\

/* 二項演算と代入 */
#define IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(OP)\
	inline This_t& operator OP##=(\
		const Temp_t &rParam\
	){\
		static_assert(Traits0_t::isValid);\
		if constexpr (Traits0_t::opeDim >= 1) { this->elements[Traits0_t::i0] OP##= rParam.elements[0]; }\
		if constexpr (Traits0_t::opeDim >= 2) { this->elements[Traits0_t::i1] OP##= rParam.elements[1]; }\
		if constexpr (Traits0_t::opeDim >= 3) { this->elements[Traits0_t::i2] OP##= rParam.elements[2]; }\
		if constexpr (Traits0_t::opeDim >= 4) { this->elements[Traits0_t::i3] OP##= rParam.elements[3]; }\
		return *this;\
	}\

/* 二項演算と代入（スカラ）*/
#define IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(OP)\
	inline This_t& operator OP##=(\
		const Element0_t param\
	){\
		static_assert(Traits0_t::isValid);\
		if constexpr (Traits0_t::opeDim >= 1) { this->elements[Traits0_t::i0] OP##= param; }\
		if constexpr (Traits0_t::opeDim >= 2) { this->elements[Traits0_t::i1] OP##= param; }\
		if constexpr (Traits0_t::opeDim >= 3) { this->elements[Traits0_t::i2] OP##= param; }\
		if constexpr (Traits0_t::opeDim >= 4) { this->elements[Traits0_t::i3] OP##= param; }\
		return *this;\
	}\

/* 二項演算（スカラ除算専用高速パス）*/
#define IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR_DIV_FAST_PATH()\
	inline Temp_t operator/(\
		const Element0_t param\
	) const {\
		Temp_t result;\
		static_assert(Traits0_t::isValid);\
		Element0_t invParam = 1 / param;\
		if constexpr (Traits0_t::opeDim >= 1) { result.elements[0] = this->elements[Traits0_t::i0] * invParam; }\
		if constexpr (Traits0_t::opeDim >= 2) { result.elements[1] = this->elements[Traits0_t::i1] * invParam; }\
		if constexpr (Traits0_t::opeDim >= 3) { result.elements[2] = this->elements[Traits0_t::i2] * invParam; }\
		if constexpr (Traits0_t::opeDim >= 4) { result.elements[3] = this->elements[Traits0_t::i3] * invParam; }\
		return result;\
	}\
\
/* 二項演算と代入（スカラ除算専用高速パス）*/
#define IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR_DIV_FAST_PATH()\
	inline This_t& operator/=(\
		const Element0_t param\
	){\
		static_assert(Traits0_t::isValid);\
		Element0_t invParam = 1 / param;\
		if constexpr (Traits0_t::opeDim >= 1) { this->elements[Traits0_t::i0] *= invParam; }\
		if constexpr (Traits0_t::opeDim >= 2) { this->elements[Traits0_t::i1] *= invParam; }\
		if constexpr (Traits0_t::opeDim >= 3) { this->elements[Traits0_t::i2] *= invParam; }\
		if constexpr (Traits0_t::opeDim >= 4) { this->elements[Traits0_t::i3] *= invParam; }\
		return *this;\
	}\

/* 二項演算（行列専用）*/
#define IMPL_VEC_BINARY_OPERATOR_WITH_MATRIX()\
	/* 変換（行列の SWIZZLE 変換を伴う）*/\
	template<int memDimC1, int memDimR1, int recursiveCount1>\
	inline This_t& operator*=(\
		const Mat<\
			Element0_t,\
			/* opeDimC1 = */ Traits0_t::opeDim, memDimC1,\
			/* opeDimR1 = */ Traits0_t::opeDim, memDimR1,\
			recursiveCount1\
		> &rParam\
	){\
		/* 入出力競合を避けるため、意図的にテンポラリオブジェクトを経由させている */\
		*this = *this * rParam;\
		return *this;\
	}\
\
	/* 変換と代入（行列の SWIZZLE 変換を伴う）*/\
	/*\
			OpeDimC1		OpeDim0		OpeDimC1\
\
										■■□□\
			■■□□	=	■■■□ *	■■□□ OpeDim0\
										■■□□\
										□□□□\
	*/\
	template<int opeDimC1, int memDimC1, int memDimR1, int recursiveCount1>\
	inline GenVec<Element0_t, opeDimC1>\
	operator*(\
		const Mat<\
			Element0_t,\
			opeDimC1, memDimC1,\
			Traits0_t::opeDim, memDimR1,\
			recursiveCount1\
		> &rParam\
	) const {\
		GenVec<Element0_t, opeDimC1> result;\
		static_assert(Traits0_t::isValid);\
		if constexpr (opeDimC1 >= 1) {\
			Element0_t tmp;\
			if constexpr (Traits0_t::opeDim >= 1) { tmp =  this->elements[Traits0_t::i0] * rParam.columns[0][0]; }\
			if constexpr (Traits0_t::opeDim >= 2) { tmp += this->elements[Traits0_t::i1] * rParam.columns[0][1]; }\
			if constexpr (Traits0_t::opeDim >= 3) { tmp += this->elements[Traits0_t::i2] * rParam.columns[0][2]; }\
			if constexpr (Traits0_t::opeDim >= 4) { tmp += this->elements[Traits0_t::i3] * rParam.columns[0][3]; }\
			result.elements[0] = tmp;\
		}\
		if constexpr (opeDimC1 >= 2) {\
			Element0_t tmp;\
			if constexpr (Traits0_t::opeDim >= 1) { tmp =  this->elements[Traits0_t::i0] * rParam.columns[1][0]; }\
			if constexpr (Traits0_t::opeDim >= 2) { tmp += this->elements[Traits0_t::i1] * rParam.columns[1][1]; }\
			if constexpr (Traits0_t::opeDim >= 3) { tmp += this->elements[Traits0_t::i2] * rParam.columns[1][2]; }\
			if constexpr (Traits0_t::opeDim >= 4) { tmp += this->elements[Traits0_t::i3] * rParam.columns[1][3]; }\
			result.elements[1] = tmp;\
		}\
		if constexpr (opeDimC1 >= 3) {\
			Element0_t tmp;\
			if constexpr (Traits0_t::opeDim >= 1) { tmp =  this->elements[Traits0_t::i0] * rParam.columns[2][0]; }\
			if constexpr (Traits0_t::opeDim >= 2) { tmp += this->elements[Traits0_t::i1] * rParam.columns[2][1]; }\
			if constexpr (Traits0_t::opeDim >= 3) { tmp += this->elements[Traits0_t::i2] * rParam.columns[2][2]; }\
			if constexpr (Traits0_t::opeDim >= 4) { tmp += this->elements[Traits0_t::i3] * rParam.columns[2][3]; }\
			result.elements[2] = tmp;\
		}\
		if constexpr (opeDimC1 >= 4) {\
			Element0_t tmp;\
			if constexpr (Traits0_t::opeDim >= 1) { tmp =  this->elements[Traits0_t::i0] * rParam.columns[3][0]; }\
			if constexpr (Traits0_t::opeDim >= 2) { tmp += this->elements[Traits0_t::i1] * rParam.columns[3][1]; }\
			if constexpr (Traits0_t::opeDim >= 3) { tmp += this->elements[Traits0_t::i2] * rParam.columns[3][2]; }\
			if constexpr (Traits0_t::opeDim >= 4) { tmp += this->elements[Traits0_t::i3] * rParam.columns[3][3]; }\
			result.elements[3] = tmp;\
		}\
		return result;\
	}\

/* インクリメント & デクリメント */
#define IMPL_VEC_INC_DEC()\
	/* 前置 ++ */\
	inline Temp_t & operator ++() {\
		static_assert(Traits0_t::isValid);\
		if constexpr (Traits0_t::opeDim >= 1) { ++this->elements[Traits0_t::i0]; }\
		if constexpr (Traits0_t::opeDim >= 2) { ++this->elements[Traits0_t::i1]; }\
		if constexpr (Traits0_t::opeDim >= 3) { ++this->elements[Traits0_t::i2]; }\
		if constexpr (Traits0_t::opeDim >= 4) { ++this->elements[Traits0_t::i3]; }\
		return *this;\
	}\
\
	/* 後置 ++ */\
	inline Temp_t operator ++(int) {\
		Temp_t result = *this;\
		static_assert(Traits0_t::isValid);\
		if constexpr (Traits0_t::opeDim >= 1) { ++this->elements[Traits0_t::i0]; }\
		if constexpr (Traits0_t::opeDim >= 2) { ++this->elements[Traits0_t::i1]; }\
		if constexpr (Traits0_t::opeDim >= 3) { ++this->elements[Traits0_t::i2]; }\
		if constexpr (Traits0_t::opeDim >= 4) { ++this->elements[Traits0_t::i3]; }\
		return result;\
	}\
\
	/* 前置 -- */\
	inline Temp_t & operator --() {\
		static_assert(Traits0_t::isValid);\
		if constexpr (Traits0_t::opeDim >= 1) { --this->elements[Traits0_t::i0]; }\
		if constexpr (Traits0_t::opeDim >= 2) { --this->elements[Traits0_t::i1]; }\
		if constexpr (Traits0_t::opeDim >= 3) { --this->elements[Traits0_t::i2]; }\
		if constexpr (Traits0_t::opeDim >= 4) { --this->elements[Traits0_t::i3]; }\
		return *this;\
	}\
\
	/* 後置 ++ */\
	inline Temp_t operator --(int) {\
		Temp_t result = *this;\
		static_assert(Traits0_t::isValid);\
		if constexpr (Traits0_t::opeDim >= 1) { --this->elements[Traits0_t::i0]; }\
		if constexpr (Traits0_t::opeDim >= 2) { --this->elements[Traits0_t::i1]; }\
		if constexpr (Traits0_t::opeDim >= 3) { --this->elements[Traits0_t::i2]; }\
		if constexpr (Traits0_t::opeDim >= 4) { --this->elements[Traits0_t::i3]; }\
		return result;\
	}\

/* 比較演算子 */
#define IMPL_VEC_COMPARISON_OPERATOR()\
	/* 比較メソッド */\
	inline int Compare(\
		const Temp_t &rParam\
	) const {\
		static_assert(Traits0_t::isValid);\
		if constexpr (Traits0_t::opeDim >= 1) {\
			if (this->elements[Traits0_t::i0] < rParam.elements[0]) { return -1; }\
			if (this->elements[Traits0_t::i0] > rParam.elements[0]) { return 1; }\
		}\
		if constexpr (Traits0_t::opeDim >= 2) {\
			if (this->elements[Traits0_t::i1] < rParam.elements[1]) { return -1; }\
			if (this->elements[Traits0_t::i1] > rParam.elements[1]) { return 1; }\
		}\
		if constexpr (Traits0_t::opeDim >= 3) {\
			if (this->elements[Traits0_t::i2] < rParam.elements[2]) { return -1; }\
			if (this->elements[Traits0_t::i2] > rParam.elements[2]) { return 1; }\
		}\
		if constexpr (Traits0_t::opeDim >= 4) {\
			if (this->elements[Traits0_t::i3] < rParam.elements[3]) { return -1; }\
			if (this->elements[Traits0_t::i3] > rParam.elements[3]) { return 1; }\
		}\
		return 0;\
	}\
\
	/* 比較演算子のバリエーション */\
	/*\
		< > <= >= は GLSL には存在しない。\
		辞書比較ルールで比較を行う。\
	*/\
	inline bool operator==(const Temp_t &rParam) const { return (Compare(rParam) == 0); };\
	inline bool operator!=(const Temp_t &rParam) const { return (Compare(rParam) != 0); };\
	inline bool operator< (const Temp_t &rParam) const { return (Compare(rParam) <  0); };\
	inline bool operator> (const Temp_t &rParam) const { return (Compare(rParam) >  0); };\
	inline bool operator<=(const Temp_t &rParam) const { return (Compare(rParam) <= 0); };\
	inline bool operator>=(const Temp_t &rParam) const { return (Compare(rParam) >= 0); };\

/* 代入演算子 */
#define IMPL_VEC_ASSIGNMENT_OPERATOR()\
	inline This_t& operator=(const This_t &rParam){\
		static_assert(Traits0_t::isValid);\
		static_assert(Traits0_t::hasDuplicatedMember == 0);\
		if constexpr (SimdVecTraits_t::isM128 && Traits0_t::opeDim == 4) {\
			__m128 tmp = rParam.simdVec;\
			this->simdVec = tmp;\
		} else\
		if constexpr (SimdVecTraits_t::isM256 && Traits0_t::opeDim == 4) {\
			__m256 tmp = rParam.simdVec;\
			this->simdVec = tmp;\
		} else {\
			if constexpr (Traits0_t::opeDim >= 1) { this->elements[Traits0_t::i0] = rParam.elements[Traits0_t::i0]; }\
			if constexpr (Traits0_t::opeDim >= 2) { this->elements[Traits0_t::i1] = rParam.elements[Traits0_t::i1]; }\
			if constexpr (Traits0_t::opeDim >= 3) { this->elements[Traits0_t::i2] = rParam.elements[Traits0_t::i2]; }\
			if constexpr (Traits0_t::opeDim >= 4) { this->elements[Traits0_t::i3] = rParam.elements[Traits0_t::i3]; }\
		}\
		return *this;\
	}\

/* キャスト */
#define IMPL_VEC_IMPLICIT_CONVERSION(Type_t, recursiveCount0)\
	/* キャスト（暗黙に行われる次元数の一致する配列から変換）*/\
	inline Type_t<Element0_t, Traits0_t, recursiveCount0>(\
		const Element0_t aParam[Traits0_t::opeDim]\
	){\
		static_assert(Traits0_t::isValid);\
		static_assert(Traits0_t::hasDuplicatedMember == 0);\
		if constexpr (Traits0_t::opeDim >= 1) { this->elements[Traits0_t::i0] = aParam[0]; }\
		if constexpr (Traits0_t::opeDim >= 2) { this->elements[Traits0_t::i1] = aParam[1]; }\
		if constexpr (Traits0_t::opeDim >= 3) { this->elements[Traits0_t::i2] = aParam[2]; }\
		if constexpr (Traits0_t::opeDim >= 4) { this->elements[Traits0_t::i3] = aParam[3]; }\
	}\
\
	/* キャスト（暗黙に行われる成分型と SWIZZLE の変換）*/\
	template<typename Element1_t, int memDim1, int swizzle1, int recursiveCount1>\
	inline Type_t<Element0_t, Traits0_t, recursiveCount0>(\
		const Type_t<\
			Element1_t,\
			Traits<\
				/* int opeDim_ */	Traits0_t::opeDim,\
				/* int memDim_ */	memDim1,\
				/* int swizzle_ */	swizzle1\
			>,\
			recursiveCount1\
		> &rParam\
	){\
		using Traits1_t = Traits<\
			/* int opeDim_ */	Traits0_t::opeDim,\
			/* int memDim_ */	memDim1,\
			/* int swizzle_ */	swizzle1\
		>;\
		static_assert(Traits0_t::isValid && Traits1_t::isValid);\
		static_assert(Traits0_t::hasDuplicatedMember == 0);\
		if constexpr (Traits0_t::opeDim >= 1) { this->elements[Traits0_t::i0] = rParam.elements[Traits1_t::i0]; }\
		if constexpr (Traits0_t::opeDim >= 2) { this->elements[Traits0_t::i1] = rParam.elements[Traits1_t::i1]; }\
		if constexpr (Traits0_t::opeDim >= 3) { this->elements[Traits0_t::i2] = rParam.elements[Traits1_t::i2]; }\
		if constexpr (Traits0_t::opeDim >= 4) { this->elements[Traits0_t::i3] = rParam.elements[Traits1_t::i3]; }\
	}\
\
	/* キャスト（暗黙に行われるスカラからのキャスト）*/\
	inline Type_t<Element0_t, Traits0_t, recursiveCount0>(\
		const Element0_t param\
	){\
		static_assert(Traits0_t::isValid);\
		static_assert(Traits0_t::hasDuplicatedMember == 0);\
		if constexpr (Traits0_t::opeDim >= 1) { this->elements[Traits0_t::i0] = param; }\
		if constexpr (Traits0_t::opeDim >= 2) { this->elements[Traits0_t::i1] = param; }\
		if constexpr (Traits0_t::opeDim >= 3) { this->elements[Traits0_t::i2] = param; }\
		if constexpr (Traits0_t::opeDim >= 4) { this->elements[Traits0_t::i3] = param; }\
	}\

/* コンストラクタ & デストラクタ */
#define IMPL_VEC_CTOR_DTOR(Type_t, recursiveCount0)\
	/* コピーコンストラクタ */\
	inline Type_t<Element0_t, Traits0_t, recursiveCount0>(\
		const This_t &rParam\
	){\
		static_assert(Traits0_t::isValid);\
		static_assert(Traits0_t::hasDuplicatedMember == 0);\
		if constexpr (Traits0_t::opeDim >= 1) { this->elements[Traits0_t::i0] = rParam.elements[Traits0_t::i0]; }\
		if constexpr (Traits0_t::opeDim >= 2) { this->elements[Traits0_t::i1] = rParam.elements[Traits0_t::i1]; }\
		if constexpr (Traits0_t::opeDim >= 3) { this->elements[Traits0_t::i2] = rParam.elements[Traits0_t::i2]; }\
		if constexpr (Traits0_t::opeDim >= 4) { this->elements[Traits0_t::i3] = rParam.elements[Traits0_t::i3]; }\
	}\
\
	/* コンストラクタ（4 次元）*/\
	inline Type_t<Element0_t, Traits0_t, recursiveCount0>(\
		const Element0_t param0,\
		const Element0_t param1,\
		const Element0_t param2,\
		const Element0_t param3\
	){\
		static_assert(Traits0_t::isValid);\
		static_assert(Traits0_t::hasDuplicatedMember == 0);\
		static_assert(Traits0_t::opeDim == 4);\
		this->elements[Traits0_t::i0] = param0;\
		this->elements[Traits0_t::i1] = param1;\
		this->elements[Traits0_t::i2] = param2;\
		this->elements[Traits0_t::i3] = param3;\
	}\
\
	/* コンストラクタ（3 次元）*/\
	inline Type_t<Element0_t, Traits0_t, recursiveCount0>(\
		const Element0_t param0,\
		const Element0_t param1,\
		const Element0_t param2\
	){\
		static_assert(Traits0_t::isValid);\
		static_assert(Traits0_t::hasDuplicatedMember == 0);\
		static_assert(Traits0_t::opeDim == 3);\
		this->elements[Traits0_t::i0] = param0;\
		this->elements[Traits0_t::i1] = param1;\
		this->elements[Traits0_t::i2] = param2;\
	}\
\
	/* コンストラクタ（2 次元）*/\
	inline Type_t<Element0_t, Traits0_t, recursiveCount0>(\
		const Element0_t param0,\
		const Element0_t param1\
	){\
		static_assert(Traits0_t::isValid);\
		static_assert(Traits0_t::hasDuplicatedMember == 0);\
		static_assert(Traits0_t::opeDim == 2);\
		this->elements[Traits0_t::i0] = param0;\
		this->elements[Traits0_t::i1] = param1;\
	}\
\
	/* コンストラクタ vec4(vec3(0,1,2),3) */\
	inline Type_t<Element0_t, Traits0_t, recursiveCount0>(\
		const Gen##Type_t<Element0_t, 3> &rParam012,\
		const Element0_t param3\
	){\
		static_assert(Traits0_t::isValid);\
		static_assert(Traits0_t::hasDuplicatedMember == 0);\
		static_assert(Traits0_t::opeDim == 4);\
		this->elements[Traits0_t::i0] = rParam012.elements[0];\
		this->elements[Traits0_t::i1] = rParam012.elements[1];\
		this->elements[Traits0_t::i2] = rParam012.elements[2];\
		this->elements[Traits0_t::i3] = param3;\
	}\
\
	/* コンストラクタ vec4(0,vec3(1,2,3)) */\
	inline Type_t<Element0_t, Traits0_t, recursiveCount0>(\
		const Element0_t param0,\
		const Gen##Type_t<Element0_t, 3> &rParam123\
	){\
		static_assert(Traits0_t::isValid);\
		static_assert(Traits0_t::hasDuplicatedMember == 0);\
		static_assert(Traits0_t::opeDim == 4);\
		this->elements[Traits0_t::i0] = param0;\
		this->elements[Traits0_t::i1] = rParam123.elements[0];\
		this->elements[Traits0_t::i2] = rParam123.elements[1];\
		this->elements[Traits0_t::i3] = rParam123.elements[2];\
	}\
\
	/* コンストラクタ vec4(vec3(0,1),2,3) */\
	inline Type_t<Element0_t, Traits0_t, recursiveCount0>(\
		const Gen##Type_t<Element0_t, 2> &rParam01,\
		const Element0_t param2,\
		const Element0_t param3\
	){\
		static_assert(Traits0_t::isValid);\
		static_assert(Traits0_t::hasDuplicatedMember == 0);\
		static_assert(Traits0_t::opeDim == 4);\
		this->elements[Traits0_t::i0] = rParam01.elements[0];\
		this->elements[Traits0_t::i1] = rParam01.elements[1];\
		this->elements[Traits0_t::i2] = param2;\
		this->elements[Traits0_t::i3] = param3;\
	}\
\
	/* コンストラクタ vec4(0,vec2(1,2),3) */\
	inline Type_t<Element0_t, Traits0_t, recursiveCount0>(\
		const Element0_t param0,\
		const Gen##Type_t<Element0_t, 2> &rParam12,\
		const Element0_t param3\
	){\
		static_assert(Traits0_t::isValid);\
		static_assert(Traits0_t::hasDuplicatedMember == 0);\
		static_assert(Traits0_t::opeDim == 4);\
		this->elements[Traits0_t::i0] = param0;\
		this->elements[Traits0_t::i1] = rParam12.elements[0];\
		this->elements[Traits0_t::i2] = rParam12.elements[1];\
		this->elements[Traits0_t::i3] = param3;\
	}\
\
	/* コンストラクタ vec4(0,1,vec2(2,3)) */\
	inline Type_t<Element0_t, Traits0_t, recursiveCount0>(\
		const Element0_t param0,\
		const Element0_t param1,\
		const Gen##Type_t<Element0_t, 2> &rParam23\
	){\
		static_assert(Traits0_t::isValid);\
		static_assert(Traits0_t::hasDuplicatedMember == 0);\
		static_assert(Traits0_t::opeDim == 4);\
		this->elements[Traits0_t::i0] = param0;\
		this->elements[Traits0_t::i1] = param1;\
		this->elements[Traits0_t::i2] = rParam23.elements[0];\
		this->elements[Traits0_t::i3] = rParam23.elements[1];\
	}\
\
	/* コンストラクタ vec4(vec2(0,1),vec2(2,3)) */\
	inline Type_t<Element0_t, Traits0_t, recursiveCount0>(\
		const Gen##Type_t<Element0_t, 2> &rParam01,\
		const Gen##Type_t<Element0_t, 2> &rParam23\
	){\
		static_assert(Traits0_t::isValid);\
		static_assert(Traits0_t::hasDuplicatedMember == 0);\
		static_assert(Traits0_t::opeDim == 4);\
		this->elements[Traits0_t::i0] = rParam01.elements[0];\
		this->elements[Traits0_t::i1] = rParam01.elements[1];\
		this->elements[Traits0_t::i2] = rParam23.elements[0];\
		this->elements[Traits0_t::i3] = rParam23.elements[1];\
	}\
\
	/* コンストラクタ vec3(vec2(0,1),2) */\
	inline Type_t<Element0_t, Traits0_t, recursiveCount0>(\
		const Gen##Type_t<Element0_t, 2> &rParam01,\
		const Element0_t param2\
	){\
		static_assert(Traits0_t::isValid);\
		static_assert(Traits0_t::hasDuplicatedMember == 0);\
		static_assert(Traits0_t::opeDim == 3);\
		this->elements[Traits0_t::i0] = rParam01.elements[0];\
		this->elements[Traits0_t::i1] = rParam01.elements[1];\
		this->elements[Traits0_t::i2] = param2;\
	}\
\
	/* コンストラクタ vec3(0,vec2(0,1)) */\
	inline Type_t<Element0_t, Traits0_t, recursiveCount0>(\
		const Element0_t param0,\
		const Gen##Type_t<Element0_t, 2> &rParam12\
	){\
		static_assert(Traits0_t::isValid);\
		static_assert(Traits0_t::hasDuplicatedMember == 0);\
		static_assert(Traits0_t::opeDim == 3);\
		this->elements[Traits0_t::i0] = param0;\
		this->elements[Traits0_t::i1] = rParam12.elements[0];\
		this->elements[Traits0_t::i2] = rParam12.elements[1];\
	}\
\
	/* コンストラクタ (次数下げ 1) */\
	explicit inline Type_t<Element0_t, Traits0_t, recursiveCount0>(\
		const Gen##Type_t<Element0_t, Traits0_t::opeDim + 1> &rParam\
	){\
		static_assert(Traits0_t::isValid);\
		static_assert(Traits0_t::hasDuplicatedMember == 0);\
		static_assert(Traits0_t::opeDim <= 3);\
		if constexpr (Traits0_t::opeDim >= 1) { this->elements[Traits0_t::i0] = rParam.elements[0]; }\
		if constexpr (Traits0_t::opeDim >= 2) { this->elements[Traits0_t::i1] = rParam.elements[1]; }\
		if constexpr (Traits0_t::opeDim >= 3) { this->elements[Traits0_t::i2] = rParam.elements[2]; }\
	}\
\
	/* コンストラクタ (次数下げ 2) */\
	explicit inline Type_t<Element0_t, Traits0_t, recursiveCount0>(\
		const Gen##Type_t<Element0_t, Traits0_t::opeDim + 2> &rParam\
	){\
		static_assert(Traits0_t::isValid);\
		static_assert(Traits0_t::hasDuplicatedMember == 0);\
		static_assert(Traits0_t::opeDim <= 2);\
		if constexpr (Traits0_t::opeDim >= 1) { this->elements[Traits0_t::i0] = rParam.elements[0]; }\
		if constexpr (Traits0_t::opeDim >= 2) { this->elements[Traits0_t::i1] = rParam.elements[1]; }\
	}\
\
	/* コンストラクタ (次数下げ 3) */\
	explicit inline Type_t<Element0_t, Traits0_t, recursiveCount0>(\
		const Gen##Type_t<Element0_t, Traits0_t::opeDim + 3> &rParam\
	){\
		static_assert(Traits0_t::isValid);\
		static_assert(Traits0_t::hasDuplicatedMember == 0);\
		static_assert(Traits0_t::opeDim <= 1);\
		if constexpr (Traits0_t::opeDim >= 1) { this->elements[Traits0_t::i0] = rParam.elements[0]; }\
	}\
\
	/* デフォルトコンストラクタ */\
	inline Type_t<Element0_t, Traits0_t, recursiveCount0>(\
	){\
	}\
\
	/* デストラクタ */\
	inline ~Type_t<Element0_t, Traits0_t, recursiveCount0>(\
	){\
	}\

/* 非メンバオペレータ */
#define IMPL_VEC_NON_MEMBER_BINARY_OPERATOR(Type_t, OP)\
	/* スカラとの乗算：二項（第一引数がスカラの場合）*/\
	template<typename Element_t, int opeDim, int memDim1, int swizzle1, int recursiveCount1>\
	static inline Gen##Type_t<Element_t, opeDim>\
	operator OP(\
		const Element_t param0,\
		const Type_t<Element_t, Traits<opeDim, memDim1, swizzle1>, recursiveCount1> &rParam1\
	){\
		using Traits1_t = Traits<opeDim, memDim1, swizzle1>;\
		Gen##Type_t<Element_t, opeDim> result;\
		if constexpr (opeDim >= 1) { result.elements[0] = param0 OP rParam1.elements[Traits1_t::i0]; }\
		if constexpr (opeDim >= 2) { result.elements[1] = param0 OP rParam1.elements[Traits1_t::i1]; }\
		if constexpr (opeDim >= 3) { result.elements[2] = param0 OP rParam1.elements[Traits1_t::i2]; }\
		if constexpr (opeDim >= 4) { result.elements[3] = param0 OP rParam1.elements[Traits1_t::i3]; }\
		return result;\
	}\
\
	/* スカラとの乗算：二項（第一引数が ScalarInVec の場合）*/\
	template<typename Element_t, int memDim0, int index0, int opeDim, int memDim1, int swizzle1, int recursiveCount1>\
	static inline Gen##Type_t<Element_t, opeDim>\
	operator OP(\
		const ScalarInVec<Element_t, memDim0, index0> &rParam0,\
		const Type_t<Element_t, Traits<opeDim, memDim1, swizzle1>, recursiveCount1> &rParam1\
	){\
		using Traits1_t = Traits<opeDim, memDim1, swizzle1>;\
		Gen##Type_t<Element_t, opeDim> result;\
		if constexpr (opeDim >= 1) { result.elements[0] = rParam0 OP rParam1.elements[Traits1_t::i0]; }\
		if constexpr (opeDim >= 2) { result.elements[1] = rParam0 OP rParam1.elements[Traits1_t::i1]; }\
		if constexpr (opeDim >= 3) { result.elements[2] = rParam0 OP rParam1.elements[Traits1_t::i2]; }\
		if constexpr (opeDim >= 4) { result.elements[3] = rParam0 OP rParam1.elements[Traits1_t::i3]; }\
		return result;\
	}\

/* address-of 演算子 */
#define IMPL_VEC_NON_MEMBER_ADDRESS_OF_OPERATOR(Type_t)\
	/* & 演算子 */\
	template<typename Element_t, typename Traits0_t, int recursiveCount0>\
	static inline Type_t<Element_t, Traits0_t, recursiveCount0> * operator&(\
		Type_t<Element_t, Traits0_t, recursiveCount0> &rThis\
	){\
		/* SWIZZLE 指定後のアドレス取得は illegal */\
		static_assert(recursiveCount0 == 0);\
		return rThis.GetThisPointer();\
	}\
\
	/* & 演算子（const 版）*/\
	template<typename Element_t, typename Traits0_t, int recursiveCount0>\
	static inline const Type_t<Element_t, Traits0_t, recursiveCount0> * operator&(\
		const Type_t<Element_t, Traits0_t, recursiveCount0> &rThis\
	){\
		/* SWIZZLE 指定後のアドレス取得は illegal */\
		static_assert(recursiveCount0 == 0);\
		return rThis.GetThisPointer();\
	}\


/*=============================================================================
▼	浮動小数ベクトルクラス
-----------------------------------------------------------------------------*/
template<typename Element0_t, typename Traits0_t, int recursiveCount0>
union Vec {
	/* 共通部分 */
	IMPL_VEC_COMMON(Vec);

	/* SWIZZLE アクセス用の共用体メンバ */
	#include "glslmath_swizzle_vec.inc.h"

	/* [] アクセス */
	IMPL_VEC_ARRAY_SUBSCRIPT_OPERATOR();

	/* 単項演算 */
	IMPL_VEC_UNARY_OPERATOR(+);
	IMPL_VEC_UNARY_OPERATOR(-);

	/* 二項演算 */
	IMPL_VEC_BINARY_OPERATOR(+);
	IMPL_VEC_BINARY_OPERATOR(-);
	IMPL_VEC_BINARY_OPERATOR(*);
	IMPL_VEC_BINARY_OPERATOR(/);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR(+);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR(-);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR(*);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR_DIV_FAST_PATH();

	/* 二項演算と代入 */
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(+);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(-);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(*);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(/);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(+);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(-);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(*);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR_DIV_FAST_PATH();

	/* 行列が関与する演算 */
	IMPL_VEC_BINARY_OPERATOR_WITH_MATRIX();

	/* インクリメント & デクリメント */
	IMPL_VEC_INC_DEC();

	/* 比較演算子 */
	IMPL_VEC_COMPARISON_OPERATOR();

	/* 各種代入演算子 */
	IMPL_VEC_ASSIGNMENT_OPERATOR();

	/* 各種キャスト */
	IMPL_VEC_IMPLICIT_CONVERSION(Vec, recursiveCount0);

#if 0
	/* 配列出力 */
	/*
		良い方法に思えるが、配列出力 -> 配列入力 によるコンストラクタが暴発してしまってダメだ。
	*/
	inline operator Element0_t * () {
		static_assert(Traits0_t::isValid);
		return this->elements;
	}
	inline operator const Element0_t * () const {
		static_assert(Traits0_t::isValid);
		return this->elements;
	}
#endif

	/* 各種コンストラクタ & デストラクタ */
	IMPL_VEC_CTOR_DTOR(Vec, recursiveCount0);
};

/* 再帰ストッパ */
template<typename Element0_t, typename Traits0_t>
union Vec<Element0_t, Traits0_t, 1>{
	enum { recursiveCount0 = 1 };

	/* 共通部分 */
	IMPL_VEC_COMMON(Vec);

	/* [] アクセス */
	IMPL_VEC_ARRAY_SUBSCRIPT_OPERATOR();

	/* 単項演算 */
	IMPL_VEC_UNARY_OPERATOR(+);
	IMPL_VEC_UNARY_OPERATOR(-);

	/* 二項演算 */
	IMPL_VEC_BINARY_OPERATOR(+);
	IMPL_VEC_BINARY_OPERATOR(-);
	IMPL_VEC_BINARY_OPERATOR(*);
	IMPL_VEC_BINARY_OPERATOR(/);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR(+);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR(-);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR(*);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR_DIV_FAST_PATH();

	/* 二項演算と代入 */
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(+);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(-);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(*);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(/);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(+);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(-);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(*);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR_DIV_FAST_PATH();

	/* 行列が関与する演算 */
	IMPL_VEC_BINARY_OPERATOR_WITH_MATRIX();

	/* インクリメント & デクリメント */
	IMPL_VEC_INC_DEC();

	/* 比較演算子 */
	IMPL_VEC_COMPARISON_OPERATOR();

	/* 各種代入演算子 */
	IMPL_VEC_ASSIGNMENT_OPERATOR();

	/* 各種キャスト */
	IMPL_VEC_IMPLICIT_CONVERSION(Vec, 1);

	/* 各種コンストラクタ & デストラクタ */
	IMPL_VEC_CTOR_DTOR(Vec, 1);
};

/* 非メンバ二項演算子 */
IMPL_VEC_NON_MEMBER_BINARY_OPERATOR(Vec, +);
IMPL_VEC_NON_MEMBER_BINARY_OPERATOR(Vec, -);
IMPL_VEC_NON_MEMBER_BINARY_OPERATOR(Vec, *);
IMPL_VEC_NON_MEMBER_BINARY_OPERATOR(Vec, /);

/* & 演算子 */
IMPL_VEC_NON_MEMBER_ADDRESS_OF_OPERATOR(Vec);


/*=============================================================================
▼	整数ベクトルクラス
-----------------------------------------------------------------------------*/
template<typename Element0_t, typename Traits0_t, int recursiveCount0>
union Ivec {
	/* 共通部分 */
	IMPL_VEC_COMMON(Ivec);

	/* SWIZZLE アクセス用の共用体メンバ */
	#include "glslmath_swizzle_ivec.inc.h"

	/* [] アクセス */
	IMPL_VEC_ARRAY_SUBSCRIPT_OPERATOR();

	/* 単項演算 */
	IMPL_VEC_UNARY_OPERATOR(+);
	IMPL_VEC_UNARY_OPERATOR(-);
	IMPL_VEC_UNARY_OPERATOR(~);

	/* 二項演算 */
	IMPL_VEC_BINARY_OPERATOR(+);
	IMPL_VEC_BINARY_OPERATOR(-);
	IMPL_VEC_BINARY_OPERATOR(*);
	IMPL_VEC_BINARY_OPERATOR(/);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR(+);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR(-);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR(*);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR(/);	/* Ivec では FAST_PATH は利用不可 */
	IMPL_VEC_BINARY_OPERATOR(&);
	IMPL_VEC_BINARY_OPERATOR(|);
	IMPL_VEC_BINARY_OPERATOR(^);
	IMPL_VEC_BINARY_OPERATOR(<<);
	IMPL_VEC_BINARY_OPERATOR(>>);

	/* 二項演算と代入 */
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(+);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(-);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(*);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(/);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(+);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(-);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(*);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(/);	/* Ivec では FAST_PATH は利用不可 */
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(&);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(|);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(^);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(<<);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(>>);

	/* 行列が関与する演算 */
	IMPL_VEC_BINARY_OPERATOR_WITH_MATRIX();

	/* インクリメント & デクリメント */
	IMPL_VEC_INC_DEC();

	/* 比較演算子 */
	IMPL_VEC_COMPARISON_OPERATOR();

	/* 各種代入演算子 */
	IMPL_VEC_ASSIGNMENT_OPERATOR();

	/* 各種キャスト */
	IMPL_VEC_IMPLICIT_CONVERSION(Ivec, recursiveCount0);

#if 0
	/* 配列出力 */
	/*
		良い方法に思えるが、配列出力 -> 配列入力 によるコンストラクタが暴発してしまってダメだ。
	*/
	inline operator Element0_t * () {
		static_assert(Traits0_t::isValid);
		return this->elements;
	}
	inline operator const Element0_t * () const {
		static_assert(Traits0_t::isValid);
		return this->elements;
	}
#endif

	/* 各種コンストラクタ & デストラクタ */
	IMPL_VEC_CTOR_DTOR(Ivec, recursiveCount0);
};

/* 再帰ストッパ */
template<typename Element0_t, typename Traits0_t>
union Ivec<Element0_t, Traits0_t, 1>{
	enum { recursiveCount0 = 1 };

	/* 共通部分 */
	IMPL_VEC_COMMON(Ivec);

	/* [] アクセス */
	IMPL_VEC_ARRAY_SUBSCRIPT_OPERATOR();

	/* 単項演算 */
	IMPL_VEC_UNARY_OPERATOR(+);
	IMPL_VEC_UNARY_OPERATOR(-);
	IMPL_VEC_UNARY_OPERATOR(~);

	/* 二項演算 */
	IMPL_VEC_BINARY_OPERATOR(+);
	IMPL_VEC_BINARY_OPERATOR(-);
	IMPL_VEC_BINARY_OPERATOR(*);
	IMPL_VEC_BINARY_OPERATOR(/);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR(+);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR(-);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR(*);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR(/);	/* Ivec では FAST_PATH は利用不可 */
	IMPL_VEC_BINARY_OPERATOR(&);
	IMPL_VEC_BINARY_OPERATOR(|);
	IMPL_VEC_BINARY_OPERATOR(^);
	IMPL_VEC_BINARY_OPERATOR(<<);
	IMPL_VEC_BINARY_OPERATOR(>>);

	/* 二項演算と代入 */
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(+);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(-);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(*);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(/);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(+);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(-);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(*);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(/);	/* Ivec では FAST_PATH は利用不可 */
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(&);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(|);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(^);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(<<);
	IMPL_VEC_COMPOUND_ASSIGNMENT_OPERATOR(>>);

	/* 行列が関与する演算 */
	IMPL_VEC_BINARY_OPERATOR_WITH_MATRIX();

	/* インクリメント & デクリメント */
	IMPL_VEC_INC_DEC();

	/* 比較演算子 */
	IMPL_VEC_COMPARISON_OPERATOR();

	/* 各種代入演算子 */
	IMPL_VEC_ASSIGNMENT_OPERATOR();

	/* 各種キャスト */
	IMPL_VEC_IMPLICIT_CONVERSION(Ivec, 1);

	/* 各種コンストラクタ & デストラクタ */
	IMPL_VEC_CTOR_DTOR(Ivec, 1);
};

/* 非メンバ二項演算子 */
IMPL_VEC_NON_MEMBER_BINARY_OPERATOR(Ivec, +);
IMPL_VEC_NON_MEMBER_BINARY_OPERATOR(Ivec, -);
IMPL_VEC_NON_MEMBER_BINARY_OPERATOR(Ivec, *);
IMPL_VEC_NON_MEMBER_BINARY_OPERATOR(Ivec, /);
IMPL_VEC_NON_MEMBER_BINARY_OPERATOR(Ivec, &);
IMPL_VEC_NON_MEMBER_BINARY_OPERATOR(Ivec, |);
IMPL_VEC_NON_MEMBER_BINARY_OPERATOR(Ivec, ^);
IMPL_VEC_NON_MEMBER_BINARY_OPERATOR(Ivec, <<);
IMPL_VEC_NON_MEMBER_BINARY_OPERATOR(Ivec, >>);

/* & 演算子 */
IMPL_VEC_NON_MEMBER_ADDRESS_OF_OPERATOR(Ivec);


/*=============================================================================
▼	bool ベクトルクラス
-----------------------------------------------------------------------------*/
/*
	bool ベクトル型 の Element0_t は、bool 型であることは自明である。
	しかし、この Element0_t は再帰ストッパの意味があるので省略できない。
*/
template<typename Element0_t, typename Traits0_t, int recursiveCount0>
union Bvec {
	/* 共通部分 */
	IMPL_VEC_COMMON(Bvec);

	/* SWIZZLE アクセス用の共用体メンバ */
	#include "glslmath_swizzle_bvec.inc.h"

	/* [] アクセス */
	IMPL_VEC_ARRAY_SUBSCRIPT_OPERATOR();

	/* 単項演算 */
	IMPL_VEC_UNARY_OPERATOR(!);

	/* 二項演算 */
	IMPL_VEC_BINARY_OPERATOR(&&);
	IMPL_VEC_BINARY_OPERATOR(||);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR(&&);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR(||);

	/* 比較演算子 */
	IMPL_VEC_COMPARISON_OPERATOR();

	/* 各種代入演算子 */
	IMPL_VEC_ASSIGNMENT_OPERATOR();

	/* キャスト */
	IMPL_VEC_IMPLICIT_CONVERSION(Bvec, recursiveCount0);

	/* 各種コンストラクタ & デストラクタ */
	IMPL_VEC_CTOR_DTOR(Bvec, recursiveCount0);
};

/* 再帰ストッパ */
template<typename Element0_t, typename Traits0_t>
union Bvec<Element0_t, Traits0_t, 1>{
	enum { recursiveCount0 = 1 };

	/* 共通部分 */
	IMPL_VEC_COMMON(Bvec);

	/* [] アクセス */
	IMPL_VEC_ARRAY_SUBSCRIPT_OPERATOR();

	/* 単項演算 */
	IMPL_VEC_UNARY_OPERATOR(!);

	/* 二項演算 */
	IMPL_VEC_BINARY_OPERATOR(&&);
	IMPL_VEC_BINARY_OPERATOR(||);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR(&&);
	IMPL_VEC_BINARY_OPERATOR_WITH_SCALAR(||);

	/* 比較演算子 */
	IMPL_VEC_COMPARISON_OPERATOR();

	/* 各種代入演算子 */
	IMPL_VEC_ASSIGNMENT_OPERATOR();

	/* キャスト */
	IMPL_VEC_IMPLICIT_CONVERSION(Bvec, 1);

	/* 各種コンストラクタ & デストラクタ */
	IMPL_VEC_CTOR_DTOR(Bvec, 1);
};

/* 非メンバ二項演算子 */
IMPL_VEC_NON_MEMBER_BINARY_OPERATOR(Bvec, &&);
IMPL_VEC_NON_MEMBER_BINARY_OPERATOR(Bvec, ||);


/*=============================================================================
▼	行列クラスコード共通部分
-----------------------------------------------------------------------------*/
/* 冒頭共通部分 */
#define IMPL_MAT_COMMON(Type_t)\
	private:\
		/* 自分自身の型 */\
		using This_t = Mat<\
			Element0_t,\
			opeDimC0, memDimC0,\
			opeDimR0, memDimR0,\
			recursiveCount0\
		>;\
\
		/* 列の型 */\
		using Column_t = GenVec2<Element0_t, opeDimR0, memDimR0>;\
\
		/* 中間値の型（SWIZZLE 解決済み）*/\
		using Temp_t = GenMat<Element0_t, opeDimC0, opeDimR0>;\
\
	public:\
		/* 列成分 */\
		Column_t columns[memDimC0];\
\
		/* this ポインタを取得 */\
		inline       This_t * GetThisPointer()       { return this; }\
		inline const This_t * GetThisPointer() const { return this; }\

/* [] アクセス */
#define IMPL_MAT_ARRAY_SUBSCRIPT_OPERATOR()\
	/* 非 const アクセス */\
	inline Column_t& operator[](unsigned int iRow){\
		/* SWIZZLE 指定後の [] アクセスは illegal */\
		static_assert(recursiveCount0 == 0);\
\
		/* SWIZZLE を無視したアクセス */\
		return this->columns[iRow];\
	}\
\
	/* const アクセス */\
	inline const Column_t& operator[](unsigned int iRow) const {\
		/* SWIZZLE 指定後の [] アクセスは illegal */\
		static_assert(recursiveCount0 == 0);\
\
		/* SWIZZLE を無視したアクセス */\
		return this->columns[iRow];\
	}\

/* 単項演算 */
#define IMPL_MAT_UNARY_OPERATOR(OP)\
	inline Temp_t operator OP() const {\
		Temp_t result;\
		if constexpr (opeDimC0 >= 1) { result.columns[0] = OP this->columns[0]; }\
		if constexpr (opeDimC0 >= 2) { result.columns[1] = OP this->columns[1]; }\
		if constexpr (opeDimC0 >= 3) { result.columns[2] = OP this->columns[2]; }\
		if constexpr (opeDimC0 >= 4) { result.columns[3] = OP this->columns[3]; }\
		return result;\
	}\

/* 二項演算 */
#define IMPL_MAT_BINARY_OPERATOR(OP)\
	inline Temp_t operator OP(\
		const Temp_t &rParam\
	) const {\
		Temp_t result;\
		if constexpr (opeDimC0 >= 1) { result.columns[0] = this->columns[0] OP rParam.columns[0]; }\
		if constexpr (opeDimC0 >= 2) { result.columns[1] = this->columns[1] OP rParam.columns[1]; }\
		if constexpr (opeDimC0 >= 3) { result.columns[2] = this->columns[2] OP rParam.columns[2]; }\
		if constexpr (opeDimC0 >= 4) { result.columns[3] = this->columns[3] OP rParam.columns[3]; }\
		return result;\
	}\

/* 二項演算（スカラ）*/
#define IMPL_MAT_BINARY_OPERATOR_WITH_SCALAR(OP)\
	inline Temp_t operator OP(\
		const Element0_t param\
	) const {\
		Temp_t result;\
		if constexpr (opeDimC0 >= 1) { result.columns[0] = this->columns[0] OP param; }\
		if constexpr (opeDimC0 >= 2) { result.columns[1] = this->columns[1] OP param; }\
		if constexpr (opeDimC0 >= 3) { result.columns[2] = this->columns[2] OP param; }\
		if constexpr (opeDimC0 >= 4) { result.columns[3] = this->columns[3] OP param; }\
		return result;\
	}\

/* 二項演算と代入 */
#define IMPL_MAT_COMPOUND_ASSIGNMENT_OPERATOR(OP)\
	inline This_t& operator OP##=(\
		const Temp_t &rParam\
	){\
		if constexpr (opeDimC0 >= 1) { this->columns[0] OP##= rParam.columns[0]; }\
		if constexpr (opeDimC0 >= 2) { this->columns[1] OP##= rParam.columns[1]; }\
		if constexpr (opeDimC0 >= 3) { this->columns[2] OP##= rParam.columns[2]; }\
		if constexpr (opeDimC0 >= 4) { this->columns[3] OP##= rParam.columns[3]; }\
		return *this;\
	}\

/* 二項演算と代入（スカラ）*/
#define IMPL_MAT_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(OP)\
	inline This_t& operator OP##=(\
		const Element0_t param\
	){\
		if constexpr (opeDimC0 >= 1) { this->columns[0] OP##= param; }\
		if constexpr (opeDimC0 >= 2) { this->columns[1] OP##= param; }\
		if constexpr (opeDimC0 >= 3) { this->columns[2] OP##= param; }\
		if constexpr (opeDimC0 >= 4) { this->columns[3] OP##= param; }\
		return *this;\
	}\

/* 二項演算（スカラ除算専用高速パス）*/
#define IMPL_MAT_BINARY_OPERATOR_WITH_SCALAR_DIV_FAST_PATH()\
	inline Temp_t operator/(\
		const Element0_t param\
	) const {\
		Temp_t result;\
		Element0_t invParam = 1 / param;\
		if constexpr (opeDimC0 >= 1) { result.columns[0] = this->columns[0] * invParam; }\
		if constexpr (opeDimC0 >= 2) { result.columns[1] = this->columns[1] * invParam; }\
		if constexpr (opeDimC0 >= 3) { result.columns[2] = this->columns[2] * invParam; }\
		if constexpr (opeDimC0 >= 4) { result.columns[3] = this->columns[3] * invParam; }\
		return result;\
	}\

/* 二項演算と代入（スカラ除算専用高速パス）*/
#define IMPL_MAT_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR_DIV_FAST_PATH()\
	inline This_t& operator/=(\
		const Element0_t param\
	){\
		Element0_t invParam = 1 / param;\
		if constexpr (opeDimC0 >= 1) { this->columns[0] *= invParam; }\
		if constexpr (opeDimC0 >= 2) { this->columns[1] *= invParam; }\
		if constexpr (opeDimC0 >= 3) { this->columns[2] *= invParam; }\
		if constexpr (opeDimC0 >= 4) { this->columns[3] *= invParam; }\
		return *this;\
	}\

/* 二項演算（ベクトル専用）*/
#define IMPL_MAT_BINARY_OPERATOR_WITH_VECTOR()\
	/* 変換 */\
	/*\
							OpeDimC0\
\
			■				■■■□				■\
			■ OpeDimR0	=	■■■□ OpeDimR0	*	■ OpeDimC0\
			□				□□□□				■\
			□				□□□□				□\
	*/\
	inline GenVec<Element0_t, opeDimR0>\
	operator*(\
		const GenVec<Element0_t, opeDimC0> &rParam\
	) const {\
		GenVec<Element0_t, opeDimR0> result;\
		if constexpr (opeDimC0 >= 1) { result =  this->columns[0] * rParam.elements[0]; }\
		if constexpr (opeDimC0 >= 2) { result += this->columns[1] * rParam.elements[1]; }\
		if constexpr (opeDimC0 >= 3) { result += this->columns[2] * rParam.elements[2]; }\
		if constexpr (opeDimC0 >= 4) { result += this->columns[3] * rParam.elements[3]; }\
		return result;\
	}\

/* 二項演算（行列専用）*/
#define IMPL_MAT_BINARY_OPERATOR_WITH_MATRIX()\
	/* 変換と代入（SWIZZLE 変換を伴う）*/\
	/*\
			OpeDimC0				OpeDimC0				OpeDimC0\
\
			■■■□				■■■□				■■■□\
			■■■□ OpeDimC0	=	■■■□ OpeDimC0	*	■■■□ OpeDimC0\
			■■■□				■■■□				■■■□\
			□□□□				□□□□				□□□□\
	*/\
	template<int memDimC1, int memDimR1, int recursiveCount1>\
	inline This_t& operator*=(\
		const Mat<\
			Element0_t,\
			/* opeDimC1 = */ opeDimC0, memDimC1,\
			/* opeDimR1 = */ opeDimC0, memDimR1,\
			recursiveCount1\
		> &rParam\
	){\
		static_assert(opeDimC0 == opeDimR0);\
		/* 入出力競合を避けるため、意図的にテンポラリオブジェクトを経由させている */\
		*this = *this * rParam;\
		return *this;\
	}\
\
	/* 変換（SWIZZLE 変換を伴う）*/\
	/*\
			OpeDimC1			OpeDimC0				OpeDimC1\
\
			■■■				■■■■				■■■□\
			■■■ OpeDimR0	=	■■■■ OpeDimR0	*	■■■□ OpeDimC0\
								□□□□				■■■□\
								□□□□				■■■□\
	*/\
	template<int opeDimC1, int memDimC1, int memDimR1, int recursiveCount1>\
	inline GenMat<Element0_t, opeDimC1, opeDimR0>\
	operator*(\
		const Mat<\
			Element0_t,\
			opeDimC1, memDimC1,\
			opeDimC0, memDimR1,\
			recursiveCount1\
		> &rParam\
	) const {\
		GenMat<Element0_t, opeDimC1, opeDimR0> result;\
		static_assert(opeDimC1 <= memDimC1);	/* rParam の整合性チェック */\
		static_assert(opeDimC0 <= memDimR1);	/* rParam の整合性チェック */\
		if constexpr (opeDimC1 /* opeDimC0 としない */ >= 1) { result.columns[0] = *this * rParam.columns[0]; }\
		if constexpr (opeDimC1 /* opeDimC0 としない */ >= 2) { result.columns[1] = *this * rParam.columns[1]; }\
		if constexpr (opeDimC1 /* opeDimC0 としない */ >= 3) { result.columns[2] = *this * rParam.columns[2]; }\
		if constexpr (opeDimC1 /* opeDimC0 としない */ >= 4) { result.columns[3] = *this * rParam.columns[3]; }\
		return result;\
	}\

/* インクリメント & デクリメント */
#define IMPL_MAT_INC_DEC()\
	/* 前置 ++ */\
	inline Temp_t & operator ++() {\
		if constexpr (opeDimC0 >= 1) { ++this->columns[0]; }\
		if constexpr (opeDimC0 >= 2) { ++this->columns[1]; }\
		if constexpr (opeDimC0 >= 3) { ++this->columns[2]; }\
		if constexpr (opeDimC0 >= 4) { ++this->columns[3]; }\
		return *this;\
	}\
\
	/* 後置 ++ */\
	inline Temp_t operator ++(int) {\
		Temp_t result = *this;\
		if constexpr (opeDimC0 >= 1) { ++this->columns[0]; }\
		if constexpr (opeDimC0 >= 2) { ++this->columns[1]; }\
		if constexpr (opeDimC0 >= 3) { ++this->columns[2]; }\
		if constexpr (opeDimC0 >= 4) { ++this->columns[3]; }\
		return result;\
	}\
\
	/* 前置 -- */\
	inline Temp_t & operator --() {\
		if constexpr (opeDimC0 >= 1) { --this->columns[0]; }\
		if constexpr (opeDimC0 >= 2) { --this->columns[1]; }\
		if constexpr (opeDimC0 >= 3) { --this->columns[2]; }\
		if constexpr (opeDimC0 >= 4) { --this->columns[3]; }\
		return *this;\
	}\
\
	/* 後置 ++ */\
	inline Temp_t operator --(int) {\
		Temp_t result = *this;\
		if constexpr (opeDimC0 >= 1) { --this->columns[0]; }\
		if constexpr (opeDimC0 >= 2) { --this->columns[1]; }\
		if constexpr (opeDimC0 >= 3) { --this->columns[2]; }\
		if constexpr (opeDimC0 >= 4) { --this->columns[3]; }\
		return result;\
	}\

/* 比較演算子 */
#define IMPL_MAT_COMPARISON_OPERATOR()\
	/* 比較メソッド */\
	inline int Compare(\
		const Temp_t &rParam\
	) const {\
		if constexpr (opeDimC0 >= 1) {\
			if (this->columns[0] < rParam.columns[0]) { return -1; }\
			if (this->columns[0] > rParam.columns[0]) { return 1; }\
		}\
		if constexpr (opeDimC0 >= 2) {\
			if (this->columns[1] < rParam.columns[1]) { return -1; }\
			if (this->columns[1] > rParam.columns[1]) { return 1; }\
		}\
		if constexpr (opeDimC0 >= 3) {\
			if (this->columns[2] < rParam.columns[2]) { return -1; }\
			if (this->columns[2] > rParam.columns[2]) { return 1; }\
		}\
		if constexpr (opeDimC0 >= 4) {\
			if (this->columns[3] < rParam.columns[3]) { return -1; }\
			if (this->columns[3] > rParam.columns[3]) { return 1; }\
		}\
		return 0;\
	}\
\
	/* 比較演算子のバリエーション */\
	/*\
		< > <= >= は GLSL には存在しない。\
		辞書比較ルールで比較を行う。\
	*/\
	inline bool operator==(const Temp_t &rParam) const { return (Compare(rParam) == 0); };\
	inline bool operator!=(const Temp_t &rParam) const { return (Compare(rParam) != 0); };\
	inline bool operator< (const Temp_t &rParam) const { return (Compare(rParam) <  0); };\
	inline bool operator> (const Temp_t &rParam) const { return (Compare(rParam) >  0); };\
	inline bool operator<=(const Temp_t &rParam) const { return (Compare(rParam) <= 0); };\
	inline bool operator>=(const Temp_t &rParam) const { return (Compare(rParam) >= 0); };\

/* 代入演算子 */
#define IMPL_MAT_ASSIGNMENT_OPERATOR()\
	/* 代入（SWIZZLE 変換を伴わない）*/\
	inline This_t& operator=(const This_t &rParam){\
		if constexpr (opeDimC0 >= 1) { this->columns[0] = rParam.columns[0]; }\
		if constexpr (opeDimC0 >= 2) { this->columns[1] = rParam.columns[1]; }\
		if constexpr (opeDimC0 >= 3) { this->columns[2] = rParam.columns[2]; }\
		if constexpr (opeDimC0 >= 4) { this->columns[3] = rParam.columns[3]; }\
		return *this;\
	}\

/* キャスト */
#define IMPL_MAT_IMPLICIT_CONVERSION(recursiveCount0)\
	/* キャスト（暗黙に行われる次元数の一致する配列から変換）*/\
	inline Mat<\
		Element0_t,\
		opeDimC0, memDimC0,\
		opeDimR0, memDimR0,\
		recursiveCount0\
	>(\
		const Element0_t aaParam[opeDimC0][opeDimR0]\
	){\
		if constexpr (opeDimC0 >= 1) { this->columns[0] = aaParam[0]; }\
		if constexpr (opeDimC0 >= 2) { this->columns[1] = aaParam[1]; }\
		if constexpr (opeDimC0 >= 3) { this->columns[2] = aaParam[2]; }\
		if constexpr (opeDimC0 >= 4) { this->columns[3] = aaParam[3]; }\
	}\
\
	/* キャスト（暗黙に行われる成分型と SWIZZLE の変換）*/\
	template<typename Element1_t, int memDimC1, int memDimR1, int recursiveCount1>\
	inline Mat<\
		Element0_t,\
		opeDimC0, memDimC0,\
		opeDimR0, memDimR0,\
		recursiveCount0\
	>(\
		const Mat<\
			Element1_t,\
			opeDimC0, memDimC1,\
			opeDimR0, memDimR1,\
			recursiveCount1\
		> &rParam\
	){\
		static_assert(opeDimC0 <= memDimC1);\
		static_assert(opeDimR0 <= memDimR1);\
		if constexpr (opeDimC0 >= 1) { this->columns[0] = rParam.columns[0]; }\
		if constexpr (opeDimC0 >= 2) { this->columns[1] = rParam.columns[1]; }\
		if constexpr (opeDimC0 >= 3) { this->columns[2] = rParam.columns[2]; }\
		if constexpr (opeDimC0 >= 4) { this->columns[3] = rParam.columns[3]; }\
	}\
\
	/* キャスト（暗黙に行われるスカラからのキャスト）*/\
	inline Mat<\
		Element0_t,\
		opeDimC0, memDimC0,\
		opeDimR0, memDimR0,\
		recursiveCount0\
	>(\
		const Element0_t param\
	){\
		if constexpr (opeDimC0 >= 1) { this->columns[0] = param; }\
		if constexpr (opeDimC0 >= 2) { this->columns[1] = param; }\
		if constexpr (opeDimC0 >= 3) { this->columns[2] = param; }\
		if constexpr (opeDimC0 >= 4) { this->columns[3] = param; }\
	}\

/* コンストラクタ & デストラクタ */
#define IMPL_MAT_CTOR_DTOR(recursiveCount0)\
	/* コピーコンストラクタ */\
	inline Mat<\
		Element0_t,\
		opeDimC0, memDimC0,\
		opeDimR0, memDimR0,\
		recursiveCount0\
	>(\
		const This_t &rParam\
	){\
		if constexpr (opeDimC0 >= 1) { this->columns[0] = rParam.columns[0]; }\
		if constexpr (opeDimC0 >= 2) { this->columns[1] = rParam.columns[1]; }\
		if constexpr (opeDimC0 >= 3) { this->columns[2] = rParam.columns[2]; }\
		if constexpr (opeDimC0 >= 4) { this->columns[3] = rParam.columns[3]; }\
	}\
\
	/* コンストラクタ（4 次元）*/\
	inline Mat<\
		Element0_t,\
		opeDimC0, memDimC0,\
		opeDimR0, memDimR0,\
		recursiveCount0\
	>(\
		const GenVec<Element0_t, opeDimR0> rParam0,\
		const GenVec<Element0_t, opeDimR0> rParam1,\
		const GenVec<Element0_t, opeDimR0> rParam2,\
		const GenVec<Element0_t, opeDimR0> rParam3\
	){\
		static_assert(opeDimC0 == 4);\
		this->columns[0] = rParam0;\
		this->columns[1] = rParam1;\
		this->columns[2] = rParam2;\
		this->columns[3] = rParam3;\
	}\
\
	/* コンストラクタ（3 次元）*/\
	inline Mat<\
		Element0_t,\
		opeDimC0, memDimC0,\
		opeDimR0, memDimR0,\
		recursiveCount0\
	>(\
		const GenVec<Element0_t, opeDimR0> rParam0,\
		const GenVec<Element0_t, opeDimR0> rParam1,\
		const GenVec<Element0_t, opeDimR0> rParam2\
	){\
		static_assert(opeDimC0 == 3);\
		this->columns[0] = rParam0;\
		this->columns[1] = rParam1;\
		this->columns[2] = rParam2;\
	}\
\
	/* コンストラクタ（2 次元）*/\
	inline Mat<\
		Element0_t,\
		opeDimC0, memDimC0,\
		opeDimR0, memDimR0,\
		recursiveCount0\
	>(\
		const GenVec<Element0_t, opeDimR0> rParam0,\
		const GenVec<Element0_t, opeDimR0> rParam1\
	){\
		static_assert(opeDimC0 == 2);\
		this->columns[0] = rParam0;\
		this->columns[1] = rParam1;\
	}\
\
	/* コンストラクタ（1 次元）*/\
	inline Mat<\
		Element0_t,\
		opeDimC0, memDimC0,\
		opeDimR0, memDimR0,\
		recursiveCount0\
	>(\
		const GenVec<Element0_t, opeDimR0> rParam0\
	){\
		static_assert(opeDimC0 == 1);\
		this->columns[0] = rParam0;\
	}\
\
	/* デフォルトコンストラクタ */\
	inline Mat<\
		Element0_t,\
		opeDimC0, memDimC0,\
		opeDimR0, memDimR0,\
		recursiveCount0\
	>(\
	){\
	}\
\
	/* デストラクタ */\
	inline ~Mat<\
		Element0_t,\
		opeDimC0, memDimC0,\
		opeDimR0, memDimR0,\
		recursiveCount0\
	>(\
	){\
	}\

/* 非メンバオペレータ */
#define IMPL_MAT_NON_MEMBER_BINARY_OPERATOR(OP)\
	/* スカラとの乗算：二項（第一引数がスカラの場合）*/\
	template<typename Element_t, int opeDimC1, int memDimC1, int opeDimR1, int memDimR1, int recursiveCount1>\
	static inline GenMat<Element_t, opeDimC1, opeDimR1>\
	operator OP(\
		const Element_t param,\
		const Mat<Element_t, opeDimC1, memDimC1, opeDimR1, memDimR1, recursiveCount1> &rThis\
	){\
		GenMat<Element_t, opeDimC1, opeDimR1> result;\
		if constexpr (opeDimC1 >= 1) { result.elements[0] = param OP rThis.columns[0]; }\
		if constexpr (opeDimC1 >= 2) { result.elements[1] = param OP rThis.columns[1]; }\
		if constexpr (opeDimC1 >= 3) { result.elements[2] = param OP rThis.columns[2]; }\
		if constexpr (opeDimC1 >= 4) { result.elements[3] = param OP rThis.columns[3]; }\
		return result;\
	}\
\
	/* スカラとの乗算：二項（第一引数が ScalarInVec の場合）*/\
	template<typename Element_t, int memDim0, int index0, int opeDimC1, int memDimC1, int opeDimR1, int memDimR1, int recursiveCount1>\
	static inline GenMat<Element_t, opeDimC1, opeDimR1>\
	operator OP(\
		const ScalarInVec<Element_t, memDim0, index0> &rParam,\
		const Mat<Element_t, opeDimC1, memDimC1, opeDimR1, memDimR1, recursiveCount1> &rThis\
	){\
		GenMat<Element_t, opeDimC1, opeDimR1> result;\
		if constexpr (opeDimC1 >= 1) { result.elements[0] = rParam OP rThis.columns[0]; }\
		if constexpr (opeDimC1 >= 2) { result.elements[1] = rParam OP rThis.columns[1]; }\
		if constexpr (opeDimC1 >= 3) { result.elements[2] = rParam OP rThis.columns[2]; }\
		if constexpr (opeDimC1 >= 4) { result.elements[3] = rParam OP rThis.columns[3]; }\
		return result;\
	}\

/* address-of 演算子 */
#define IMPL_MAT_NON_MEMBER_ADDRESS_OF_OPERATOR()\
	/* address-of 演算子 */\
	template<typename Element0_t, int opeDimC0, int memDimC0, int opeDimR0, int memDimR0, int recursiveCount0>\
	static inline Mat<Element0_t, opeDimC0, memDimC0, opeDimR0, memDimR0, recursiveCount0> * operator&(\
		Mat<Element0_t, opeDimC0, memDimC0, opeDimR0, memDimR0, recursiveCount0> &rThis\
	){\
		/* SWIZZLE 指定後のアドレス取得は illegal */\
		static_assert(recursiveCount0 == 0);\
		return rThis.GetThisPointer();\
	}\
\
	/* address-of（const 版）演算子 */\
	template<typename Element0_t, int opeDimC0, int memDimC0, int opeDimR0, int memDimR0, int recursiveCount0>\
	static inline const Mat<Element0_t, opeDimC0, memDimC0, opeDimR0, memDimR0, recursiveCount0> * operator&(\
		const Mat<Element0_t, opeDimC0, memDimC0, opeDimR0, memDimR0, recursiveCount0> &rThis\
	){\
		/* SWIZZLE 指定後のアドレス取得は illegal */\
		static_assert(recursiveCount0 == 0);\
		return rThis.GetThisPointer();\
	}\


/*=============================================================================
▼	行列クラス
-----------------------------------------------------------------------------*/
template<
	typename Element0_t,
	int opeDimC0, int memDimC0,
	int opeDimR0, int memDimR0,
	int recursiveCount0
>
union Mat {
	/* 共通部分 */
	IMPL_MAT_COMMON(Mat);

	/* SWIZZLE アクセス用の共用体メンバ */
	#include "glslmath_swizzle_mat.inc.h"

	/* [] アクセス */
	IMPL_MAT_ARRAY_SUBSCRIPT_OPERATOR();

	/* 単項演算 */
	IMPL_MAT_UNARY_OPERATOR(+);
	IMPL_MAT_UNARY_OPERATOR(-);

	/* 二項演算 */
	IMPL_MAT_BINARY_OPERATOR(+);
	IMPL_MAT_BINARY_OPERATOR(-);
	IMPL_MAT_BINARY_OPERATOR_WITH_SCALAR(+);
	IMPL_MAT_BINARY_OPERATOR_WITH_SCALAR(-);
	IMPL_MAT_BINARY_OPERATOR_WITH_SCALAR(*);
	IMPL_MAT_BINARY_OPERATOR_WITH_SCALAR_DIV_FAST_PATH();

	/* 二項演算と代入 */
	IMPL_MAT_COMPOUND_ASSIGNMENT_OPERATOR(+);
	IMPL_MAT_COMPOUND_ASSIGNMENT_OPERATOR(-);
	IMPL_MAT_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(+);
	IMPL_MAT_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(-);
	IMPL_MAT_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(*);
	IMPL_MAT_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR_DIV_FAST_PATH();

	/* 行列とベクトルの演算 */
	IMPL_MAT_BINARY_OPERATOR_WITH_VECTOR();

	/* 行列同士の演算 */
	IMPL_MAT_BINARY_OPERATOR_WITH_MATRIX();

	/* インクリメント & デクリメント */
	IMPL_MAT_INC_DEC();

	/* 比較演算子 */
	IMPL_MAT_COMPARISON_OPERATOR();

	/* 各種代入演算子 */
	IMPL_MAT_ASSIGNMENT_OPERATOR();

	/* キャスト */
	IMPL_MAT_IMPLICIT_CONVERSION(recursiveCount0);

#if 0
	/* 配列出力 */
	inline operator Element0_t * () {
		return this->columns[0].elements;
	}
	inline operator const Element0_t * () const {
		return this->columns[0].elements;
	}
#endif

/* 各種コンストラクタ & デストラクタ */
	IMPL_MAT_CTOR_DTOR(recursiveCount0);
};

/* 再帰ストッパ */
template<
	typename Element0_t,
	int opeDimC0, int memDimC0,
	int opeDimR0, int memDimR0
>
union Mat<Element0_t, opeDimC0, memDimC0, opeDimR0, memDimR0, 1>{
	enum { recursiveCount0 = 1 };

	/* 共通部分 */
	IMPL_MAT_COMMON(Vec);

	/* [] アクセス */
	IMPL_MAT_ARRAY_SUBSCRIPT_OPERATOR();

	/* 単項演算 */
	IMPL_MAT_UNARY_OPERATOR(+);
	IMPL_MAT_UNARY_OPERATOR(-);

	/* 二項演算 */
	IMPL_MAT_BINARY_OPERATOR(+);
	IMPL_MAT_BINARY_OPERATOR(-);
	IMPL_MAT_BINARY_OPERATOR_WITH_SCALAR(+);
	IMPL_MAT_BINARY_OPERATOR_WITH_SCALAR(-);
	IMPL_MAT_BINARY_OPERATOR_WITH_SCALAR(*);
	IMPL_MAT_BINARY_OPERATOR_WITH_SCALAR_DIV_FAST_PATH();

	/* 二項演算と代入 */
	IMPL_MAT_COMPOUND_ASSIGNMENT_OPERATOR(+);
	IMPL_MAT_COMPOUND_ASSIGNMENT_OPERATOR(-);
	IMPL_MAT_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(+);
	IMPL_MAT_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(-);
	IMPL_MAT_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR(*);
	IMPL_MAT_COMPOUND_ASSIGNMENT_OPERATOR_WITH_SCALAR_DIV_FAST_PATH();

	/* 行列とベクトルの演算 */
	IMPL_MAT_BINARY_OPERATOR_WITH_VECTOR();

	/* 行列同士の演算 */
	IMPL_MAT_BINARY_OPERATOR_WITH_MATRIX();

	/* インクリメント & デクリメント */
	IMPL_MAT_INC_DEC();

	/* 比較演算子 */
	IMPL_MAT_COMPARISON_OPERATOR();

	/* 各種代入演算子 */
	IMPL_MAT_ASSIGNMENT_OPERATOR();

	/* 各種キャスト */
	IMPL_MAT_IMPLICIT_CONVERSION(1);

	/* 各種コンストラクタ & デストラクタ */
	IMPL_MAT_CTOR_DTOR(1);
};

/* 非メンバオペレータ */
IMPL_MAT_NON_MEMBER_BINARY_OPERATOR(+);
IMPL_MAT_NON_MEMBER_BINARY_OPERATOR(-);
IMPL_MAT_NON_MEMBER_BINARY_OPERATOR(*);
IMPL_MAT_NON_MEMBER_BINARY_OPERATOR(/);

/* address-of 演算子 */
IMPL_MAT_NON_MEMBER_ADDRESS_OF_OPERATOR();


/*=============================================================================
▼	Bvec 対応比較関数
-----------------------------------------------------------------------------*/
#define IMPL_NON_MEMBER_COMPARISON_FUNCTION(funcName, OP)\
template<typename Element_t, int opeDim, int memDim0, int memDim1, int swizzle0, int swizzle1, int recursiveCount0, int recursiveCount1>\
static inline GenBvec<bool, opeDim> funcName(\
	const Vec<Element_t, Traits<opeDim, memDim0, swizzle0>, recursiveCount0> &rParam0,\
	const Vec<Element_t, Traits<opeDim, memDim1, swizzle1>, recursiveCount1> &rParam1\
){\
	using Traits0_t = Traits<opeDim, memDim0, swizzle0>;\
	using Traits1_t = Traits<opeDim, memDim1, swizzle1>;\
	static_assert(Traits0_t::isValid);\
	static_assert(Traits1_t::isValid);\
	GenBvec<bool, opeDim> result;\
	if constexpr (opeDim >= 1) { result.elements[0] = (rParam0.elements[Traits0_t::i0] OP rParam1.elements[Traits1_t::i0]); }\
	if constexpr (opeDim >= 2) { result.elements[1] = (rParam0.elements[Traits0_t::i1] OP rParam1.elements[Traits1_t::i1]); }\
	if constexpr (opeDim >= 3) { result.elements[2] = (rParam0.elements[Traits0_t::i2] OP rParam1.elements[Traits1_t::i2]); }\
	if constexpr (opeDim >= 4) { result.elements[3] = (rParam0.elements[Traits0_t::i3] OP rParam1.elements[Traits1_t::i3]); }\
	return result;\
}\
\
template<typename Element_t, int opeDim, int memDim0, int memDim1, int swizzle0, int swizzle1, int recursiveCount0, int recursiveCount1>\
static inline GenBvec<bool, opeDim> funcName(\
	const Ivec<Element_t, Traits<opeDim, memDim0, swizzle0>, recursiveCount0> &rParam0,\
	const Ivec<Element_t, Traits<opeDim, memDim1, swizzle1>, recursiveCount1> &rParam1\
){\
	using Traits0_t = Traits<opeDim, memDim0, swizzle0>;\
	using Traits1_t = Traits<opeDim, memDim1, swizzle1>;\
	static_assert(Traits0_t::isValid);\
	static_assert(Traits1_t::isValid);\
	GenBvec<bool, opeDim> result;\
	if constexpr (opeDim >= 1) { result.elements[0] = (rParam0.elements[Traits0_t::i0] OP rParam1.elements[Traits1_t::i0]); }\
	if constexpr (opeDim >= 2) { result.elements[1] = (rParam0.elements[Traits0_t::i1] OP rParam1.elements[Traits1_t::i1]); }\
	if constexpr (opeDim >= 3) { result.elements[2] = (rParam0.elements[Traits0_t::i2] OP rParam1.elements[Traits1_t::i2]); }\
	if constexpr (opeDim >= 4) { result.elements[3] = (rParam0.elements[Traits0_t::i3] OP rParam1.elements[Traits1_t::i3]); }\
	return result;\
}\

IMPL_NON_MEMBER_COMPARISON_FUNCTION(lessThan, <);
IMPL_NON_MEMBER_COMPARISON_FUNCTION(lessThanEqual, <=);
IMPL_NON_MEMBER_COMPARISON_FUNCTION(greaterThan, >);
IMPL_NON_MEMBER_COMPARISON_FUNCTION(greaterThanEqual, >=);
IMPL_NON_MEMBER_COMPARISON_FUNCTION(equal, ==);
IMPL_NON_MEMBER_COMPARISON_FUNCTION(notEqual, !=);

template<typename Element0_t, typename Traits0_t, int recursiveCount0>
static inline bool any(
	const Bvec<Element0_t, Traits0_t, recursiveCount0> &rParam
){
	static_assert(Traits0_t::isValid);
	if constexpr (Traits0_t::opeDim >= 1) { if (rParam.elements[Traits0_t::i0]) return true; }
	if constexpr (Traits0_t::opeDim >= 2) { if (rParam.elements[Traits0_t::i1]) return true; }
	if constexpr (Traits0_t::opeDim >= 3) { if (rParam.elements[Traits0_t::i2]) return true; }
	if constexpr (Traits0_t::opeDim >= 4) { if (rParam.elements[Traits0_t::i3]) return true; }
	return false;
}

template<typename Element0_t, typename Traits0_t, int recursiveCount0>
static inline bool all(
	const Bvec<Element0_t, Traits0_t, recursiveCount0> &rParam
){
	static_assert(Traits0_t::isValid);
	if constexpr (Traits0_t::opeDim >= 1) { if (rParam.elements[Traits0_t::i0] == false) return false; }
	if constexpr (Traits0_t::opeDim >= 2) { if (rParam.elements[Traits0_t::i1] == false) return false; }
	if constexpr (Traits0_t::opeDim >= 3) { if (rParam.elements[Traits0_t::i2] == false) return false; }
	if constexpr (Traits0_t::opeDim >= 4) { if (rParam.elements[Traits0_t::i3] == false) return false; }
	return true;
}

#ifdef _MSC_VER
template<typename Element0_t, typename Traits0_t, int recursiveCount0>
static inline GenBvec<Element0_t, Traits0_t::opeDim> not(
	const Bvec<Element0_t, Traits0_t, recursiveCount0> &rParam
){
	return !rParam;
}
#else
/*
	not() は operator ! としてすでに定義済みである。
*/
#endif


/*=============================================================================
▼	関数定義
-----------------------------------------------------------------------------*/
static inline float radians(float degrees){
	return degrees * float(/*M_PI*/ 3.14159265358979323846) / 180.0f;
}
static inline double radians(double degrees){
	return degrees * double(/*M_PI*/ 3.14159265358979323846) / 180.0;
}

static inline float degrees(float radians){
	return radians * 180.0f / float(/*M_PI*/ 3.14159265358979323846);
}
static inline double degrees(double radians){
	return radians * 180.0 / double(/*M_PI*/ 3.14159265358979323846);
}

using ::std::sin;
using ::std::cos;
using ::std::tan;
using ::std::asin;
using ::std::acos;
using ::std::atan;

static inline float atan(float y, float x){
	return std::atan2(y, x);
}
static inline double atan(double y, double x){
	return std::atan2(y, x);
}

using ::std::sinh;
using ::std::cosh;
using ::std::tanh;
using ::std::asinh;
using ::std::acosh;
using ::std::atanh;

using ::std::pow;
using ::std::exp;
using ::std::log;
using ::std::exp2;
using ::std::log2;

using ::std::sqrt;

static inline float invertsqrt(float x){
	return std::sqrt(1.0f / x);
}
static inline double invertsqrt(double x){
	return std::sqrt(1.0 / x);
}

using ::std::abs;

static inline int sign(int x){
	if (x < 0) return -1;
	if (x > 0) return 1;
	return 0;
}
static inline float sign(float x){
	if (x < 0) return -1.0f;
	if (x > 0) return 1.0f;
	return 0.0f;
}
static inline double sign(double x){
	if (x < 0) return -1.0;
	if (x > 0) return 1.0;
	return 0.0;
}

using ::std::floor;
using ::std::trunc;
using ::std::round;

static inline float roundEven(float x){
	float i = std::floor(x);
	x -= i;
	if (x < 0.5f) return i;
	if (x > 0.5f) return i + 1.0f;
	float intPart;
	(void)std::modf(i / 2.0f, &intPart);
	if ((2.0f * intPart) == i) return i;
	return i + 1.0f;
}
static inline double roundEven(double x){
	double i = std::floor(x);
	x -= i;
	if (x < 0.5) return i;
	if (x > 0.5) return i + 1.0;
	double intPart;
	(void)std::modf(i / 2.0, &intPart);
	if ((2.0 * intPart) == i) return i;
	return i + 1.0;
}

using ::std::ceil;

static inline float fract(float x){
	return 1.0f - std::floor(x);
}
static inline double fract(double x){
	return 1.0 - std::floor(x);
}

static inline float mod(float x, float y){
	return std::fmod(x, y);
}
static inline double mod(double x, double y){
	return std::fmod(x, y);
}

/* 標準の modf は、第二引数はポインタだが、GLSL の仕様では参照になる。*/
static inline float modf(float x, float &y){
	return std::modf(x, &y);
}
static inline double modf(double x, double &y){
	return std::modf(x, &y);
}
template<int memDim0, int index0>
static inline float modf(float x, ScalarInVec<float, memDim0, index0> &y){
	return std::modf(x, &y);
}
template<int memDim0, int index0>
static inline double modf(double x, ScalarInVec<double, memDim0, index0> &y){
	return std::modf(x, &y);
}

#if 0
using ::std::min;			/* @@ なぜかうまくいかない */
#else
static inline float min(float x, float y){
	return x < y ? x : y;
}
static inline double min(double x, double y){
	return x < y ? x : y;
}
#endif

#if 0
using ::std::max;			/* @@ なぜかうまくいかない */
#else
static inline float max(float x, float y){
	return x > y ? x : y;
}
static inline double max(double x, double y){
	return x > y ? x : y;
}
#endif

static inline int clamp(int x, int minVal, int maxVal){
	return std::max(std::min(x, minVal), maxVal);
}
static inline float clamp(float x, float minVal, float maxVal){
	return std::max(std::min(x, minVal), maxVal);
}
static inline double clamp(double x, double minVal, double maxVal){
	return std::max(std::min(x, minVal), maxVal);
}

static inline float mix(float x, float y, float a){
//	return std::lerp(x, y, a);		/* C++20 */
	return x + (y - x) * a;
}
static inline double mix(double x, double y, double a){
//	return std::lerp(x, y, a);		/* C++20 */
	return x + (y - x) * a;
}

static inline float step(float edge, float x){
	return (x < edge)? 0.0f : 1.0f;
}
static inline double step(double edge, double x){
	return (x < edge)? 0.0 : 1.0;
}

static inline float smoothstep(float edge0, float edge1, float x){
	float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
	return t * t * (3.0f - 2.0f * t);
}
static inline double smoothstep(double edge0, double edge1, double x){
	double t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
	return t * t * (3.0 - 2.0 * t);
}

using ::std::isnan;
using ::std::isinf;

static inline int floatBitsToInt(float value){
	union {
		float asFloat;
		int asInt;
	} u;
	u.asFloat = value;
	return u.asInt;
}
static inline unsigned int floatBitsToUint(float value){
	union {
		float asFloat;
		unsigned int asUint;
	} u;
	u.asFloat = value;
	return u.asUint;
}
static inline float intBitsToFloat(int value){
	union {
		int asInt;
		float asFloat;
	} u;
	u.asInt = value;
	return u.asFloat;
}
static inline float uintBitsToFloat(unsigned int value){
	union {
		unsigned int asUint;
		float asFloat;
	} u;
	u.asUint = value;
	return u.asFloat;
}

using ::std::fma;

/* 標準の frexp は、第二引数はポインタだが、GLSL の仕様では参照になる。*/
static inline float frexp(float x, int &y){
	return std::frexp(x, &y);
}
static inline double frexp(double x, int &y){
	return std::frexp(x, &y);
}
template<int memDim0, int index0>
static inline float frexp(float x, ScalarInVec<int, memDim0, index0> &y){
	return std::frexp(x, &y);
}
template<int memDim0, int index0>
static inline double frexp(double x, ScalarInVec<int, memDim0, index0> &y){
	return std::frexp(x, &y);
}

using ::std::ldexp;


/*=============================================================================
▼	ベクトル用の関数定義
-----------------------------------------------------------------------------*/
/* 引数が 1 個の要素ごとの関数呼び出し */
#define IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(InputType_t, OutputType_t, funcName, subFunc)\
template<typename Element_t, typename Traits_t, int recursiveCount>\
static inline Gen##OutputType_t<Element_t, Traits_t::opeDim> funcName(\
	const InputType_t<Element_t, Traits_t, recursiveCount> &rParam\
){\
	static_assert(Traits_t::isValid);\
	Gen##OutputType_t<Element_t, Traits_t::opeDim> result;\
	if constexpr (Traits_t::opeDim >= 1) { result.elements[0] = subFunc(rParam.elements[Traits_t::i0]); }\
	if constexpr (Traits_t::opeDim >= 2) { result.elements[1] = subFunc(rParam.elements[Traits_t::i1]); }\
	if constexpr (Traits_t::opeDim >= 3) { result.elements[2] = subFunc(rParam.elements[Traits_t::i2]); }\
	if constexpr (Traits_t::opeDim >= 4) { result.elements[3] = subFunc(rParam.elements[Traits_t::i3]); }\
	return result;\
}\

/* 引数が 2 個の要素ごとの関数呼び出し */
#define IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_2ARGS(InputType_t, OutputType_t, funcName, subFunc)\
template<typename Element_t, int opeDim, int memDim0, int memDim1, int swizzle0, int swizzle1, int recursiveCount0, int recursiveCount1>\
static inline Gen##OutputType_t<Element_t, opeDim>\
funcName(\
	const InputType_t<Element_t, Traits<opeDim, memDim0, swizzle0>, recursiveCount0> &rParam0,\
	const InputType_t<Element_t, Traits<opeDim, memDim1, swizzle1>, recursiveCount1> &rParam1\
){\
	using Traits0_t = Traits<opeDim, memDim0, swizzle0>;\
	using Traits1_t = Traits<opeDim, memDim1, swizzle1>;\
	static_assert(Traits0_t::isValid);\
	static_assert(Traits1_t::isValid);\
	Gen##OutputType_t<Element_t, opeDim> result;\
	if constexpr (opeDim >= 1) { result.elements[0] = subFunc(rParam0.elements[Traits0_t::i0], rParam1.elements[Traits1_t::i0]); }\
	if constexpr (opeDim >= 2) { result.elements[1] = subFunc(rParam0.elements[Traits0_t::i1], rParam1.elements[Traits1_t::i1]); }\
	if constexpr (opeDim >= 3) { result.elements[2] = subFunc(rParam0.elements[Traits0_t::i2], rParam1.elements[Traits1_t::i2]); }\
	if constexpr (opeDim >= 4) { result.elements[3] = subFunc(rParam0.elements[Traits0_t::i3], rParam1.elements[Traits1_t::i3]); }\
	return result;\
}\

/* 引数が 2 個の要素ごとの関数呼び出し（2 個めの引数は結果受け取り）*/
#define IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_2ARGS_RET(InputType_t, OutputType_t, funcName, subFunc)\
template<typename Element_t, int opeDim, int memDim0, int memDim1, int swizzle0, int swizzle1, int recursiveCount0, int recursiveCount1>\
static inline Gen##OutputType_t<Element_t, opeDim>\
funcName(\
	const InputType_t<Element_t, Traits<opeDim, memDim0, swizzle0>, recursiveCount0> &rParam0,\
	      InputType_t<Element_t, Traits<opeDim, memDim1, swizzle1>, recursiveCount1> &rParam1\
){\
	using Traits0_t = Traits<opeDim, memDim0, swizzle0>;\
	using Traits1_t = Traits<opeDim, memDim1, swizzle1>;\
	static_assert(Traits0_t::isValid);\
	static_assert(Traits1_t::isValid);\
	Gen##OutputType_t<Element_t, opeDim> result;\
	if constexpr (opeDim >= 1) { result.elements[0] = subFunc(rParam0.elements[Traits0_t::i0], rParam1.elements[Traits1_t::i0]); }\
	if constexpr (opeDim >= 2) { result.elements[1] = subFunc(rParam0.elements[Traits0_t::i1], rParam1.elements[Traits1_t::i1]); }\
	if constexpr (opeDim >= 3) { result.elements[2] = subFunc(rParam0.elements[Traits0_t::i2], rParam1.elements[Traits1_t::i2]); }\
	if constexpr (opeDim >= 4) { result.elements[3] = subFunc(rParam0.elements[Traits0_t::i3], rParam1.elements[Traits1_t::i3]); }\
	return result;\
}\

/* 引数が 3 個の要素ごとの関数呼び出し */
#define IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_3ARGS(InputType_t, OutputType_t, funcName, subFunc)\
template<typename Element_t, int opeDim, int memDim0, int memDim1, int memDim2, int swizzle0, int swizzle1, int swizzle2, int recursiveCount0, int recursiveCount1, int recursiveCount2>\
static inline Gen##OutputType_t<Element_t, opeDim>\
funcName(\
	const InputType_t<Element_t, Traits<opeDim, memDim0, swizzle0>, recursiveCount0> &rParam0,\
	const InputType_t<Element_t, Traits<opeDim, memDim1, swizzle1>, recursiveCount1> &rParam1,\
	const InputType_t<Element_t, Traits<opeDim, memDim2, swizzle2>, recursiveCount2> &rParam2\
){\
	using Traits0_t = Traits<opeDim, memDim0, swizzle0>;\
	using Traits1_t = Traits<opeDim, memDim1, swizzle1>;\
	using Traits2_t = Traits<opeDim, memDim2, swizzle2>;\
	static_assert(Traits0_t::isValid);\
	static_assert(Traits1_t::isValid);\
	static_assert(Traits2_t::isValid);\
	Gen##OutputType_t<Element_t, opeDim> result;\
	if constexpr (opeDim >= 1) { result.elements[0] = subFunc(rParam0.elements[Traits0_t::i0], rParam1.elements[Traits1_t::i0], rParam2.elements[Traits2_t::i0]); }\
	if constexpr (opeDim >= 2) { result.elements[1] = subFunc(rParam0.elements[Traits0_t::i1], rParam1.elements[Traits1_t::i1], rParam2.elements[Traits2_t::i1]); }\
	if constexpr (opeDim >= 3) { result.elements[2] = subFunc(rParam0.elements[Traits0_t::i2], rParam1.elements[Traits1_t::i2], rParam2.elements[Traits2_t::i2]); }\
	if constexpr (opeDim >= 4) { result.elements[3] = subFunc(rParam0.elements[Traits0_t::i3], rParam1.elements[Traits1_t::i3], rParam2.elements[Traits2_t::i3]); }\
	return result;\
}\


IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, radians, radians);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, degrees, degrees);

IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, sin, sin);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, cos, cos);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, tan, tan);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, asin, asin);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, acos, acos);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, atan, atan);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_2ARGS(Vec, Vec, atan, atan);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, sinh, sinh);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, cosh, cosh);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, tanh, tanh);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, asinh, asinh);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, acosh, acosh);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, atanh, atanh);

IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_2ARGS(Vec, Vec, pow, pow);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, exp, exp);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, log, log);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, exp2, exp2);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, log2, log2);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, sqrt, sqrt);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, invertsqrt, invertsqrt);

IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, abs, abs);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Ivec, Ivec, abs, abs);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, sign, sign);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Ivec, Ivec, sign, sign);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, floor, floor);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, trunc, trunc);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, round, round);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, roundEven, roundEven);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, ceil, ceil);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Vec, fract, fract);

IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_2ARGS(Vec, Vec, mod, mod);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_2ARGS_RET(Vec, Vec, modf, modf);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_2ARGS(Vec, Vec, min, min);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_2ARGS(Vec, Vec, max, max);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_3ARGS(Vec, Vec, clamp, clamp);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_3ARGS(Vec, Vec, mix, mix);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_2ARGS(Vec, Vec, step, step);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_3ARGS(Vec, Vec, smoothstep, smoothstep);

IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Bvec, isnan, isnan);
IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_1ARGS(Vec, Bvec, isinf, isinf);


template<typename Traits_t, int recursiveCount>
static inline GenIvec<int, Traits_t::opeDim> floatBitsToInt(
	const Vec<float, Traits_t, recursiveCount> &rParam
){
	static_assert(Traits_t::isValid);
	GenIvec<int, Traits_t::opeDim> result;
	if constexpr (Traits_t::opeDim >= 1) { result.elements[0] = floatBitsToInt(rParam.elements[Traits_t::i0]); }
	if constexpr (Traits_t::opeDim >= 2) { result.elements[1] = floatBitsToInt(rParam.elements[Traits_t::i1]); }
	if constexpr (Traits_t::opeDim >= 3) { result.elements[2] = floatBitsToInt(rParam.elements[Traits_t::i2]); }
	if constexpr (Traits_t::opeDim >= 4) { result.elements[3] = floatBitsToInt(rParam.elements[Traits_t::i3]); }
	return result;
}

template<typename Traits_t, int recursiveCount>
static inline GenIvec<unsigned int, Traits_t::opeDim> floatBitsToUint(
	const Vec<float, Traits_t, recursiveCount> &rParam
){
	static_assert(Traits_t::isValid);
	GenIvec<unsigned int, Traits_t::opeDim> result;
	if constexpr (Traits_t::opeDim >= 1) { result.elements[0] = floatBitsToUint(rParam.elements[Traits_t::i0]); }
	if constexpr (Traits_t::opeDim >= 2) { result.elements[1] = floatBitsToUint(rParam.elements[Traits_t::i1]); }
	if constexpr (Traits_t::opeDim >= 3) { result.elements[2] = floatBitsToUint(rParam.elements[Traits_t::i2]); }
	if constexpr (Traits_t::opeDim >= 4) { result.elements[3] = floatBitsToUint(rParam.elements[Traits_t::i3]); }
	return result;
}


template<typename Traits_t, int recursiveCount>
static inline GenVec<float, Traits_t::opeDim> intBitsToFloat(
	const Ivec<int, Traits_t, recursiveCount> &rParam
){
	static_assert(Traits_t::isValid);
	GenVec<float, Traits_t::opeDim> result;
	if constexpr (Traits_t::opeDim >= 1) { result.elements[0] = intBitsToFloat(rParam.elements[Traits_t::i0]); }
	if constexpr (Traits_t::opeDim >= 2) { result.elements[1] = intBitsToFloat(rParam.elements[Traits_t::i1]); }
	if constexpr (Traits_t::opeDim >= 3) { result.elements[2] = intBitsToFloat(rParam.elements[Traits_t::i2]); }
	if constexpr (Traits_t::opeDim >= 4) { result.elements[3] = intBitsToFloat(rParam.elements[Traits_t::i3]); }
	return result;
}

template<typename Traits_t, int recursiveCount>
static inline GenVec<float, Traits_t::opeDim> uintBitsToFloat(
	const Ivec<unsigned int, Traits_t, recursiveCount> &rParam
){
	static_assert(Traits_t::isValid);
	GenVec<float, Traits_t::opeDim> result;
	if constexpr (Traits_t::opeDim >= 1) { result.elements[0] = uintBitsToFloat(rParam.elements[Traits_t::i0]); }
	if constexpr (Traits_t::opeDim >= 2) { result.elements[1] = uintBitsToFloat(rParam.elements[Traits_t::i1]); }
	if constexpr (Traits_t::opeDim >= 3) { result.elements[2] = uintBitsToFloat(rParam.elements[Traits_t::i2]); }
	if constexpr (Traits_t::opeDim >= 4) { result.elements[3] = uintBitsToFloat(rParam.elements[Traits_t::i3]); }
	return result;
}

IMPL_VEC_NON_MEMBER_ELEMENTWISE_FUNCTION_3ARGS(Vec, Vec, fma, fma);

template<typename Element_t, int opeDim, int memDim0, int memDim1, int swizzle0, int swizzle1, int recursiveCount0, int recursiveCount1>
static inline GenVec<Element_t, opeDim>
frexp(
	const Vec<Element_t, Traits<opeDim, memDim0, swizzle0>, recursiveCount0> &rParam0,
	      Ivec<int, Traits<opeDim, memDim1, swizzle1>, recursiveCount1> &rParam1
){
	using Traits0_t = Traits<opeDim, memDim0, swizzle0>;
	using Traits1_t = Traits<opeDim, memDim1, swizzle1>;
	static_assert(Traits0_t::isValid);
	static_assert(Traits1_t::isValid);
	GenVec<Element_t, opeDim> result;
	if constexpr (opeDim >= 1) { result.elements[0] = frexp(rParam0.elements[Traits0_t::i0], rParam1.elements[Traits1_t::i0]); }
	if constexpr (opeDim >= 2) { result.elements[1] = frexp(rParam0.elements[Traits0_t::i1], rParam1.elements[Traits1_t::i1]); }
	if constexpr (opeDim >= 3) { result.elements[2] = frexp(rParam0.elements[Traits0_t::i2], rParam1.elements[Traits1_t::i2]); }
	if constexpr (opeDim >= 4) { result.elements[3] = frexp(rParam0.elements[Traits0_t::i3], rParam1.elements[Traits1_t::i3]); }
	return result;
}

template<typename Element_t, int opeDim, int memDim0, int memDim1, int swizzle0, int swizzle1, int recursiveCount0, int recursiveCount1>
static inline GenVec<Element_t, opeDim>
ldexp(
	const Vec<Element_t, Traits<opeDim, memDim0, swizzle0>, recursiveCount0> &rParam0,
	      Ivec<int, Traits<opeDim, memDim1, swizzle1>, recursiveCount1> &rParam1
){
	using Traits0_t = Traits<opeDim, memDim0, swizzle0>;
	using Traits1_t = Traits<opeDim, memDim1, swizzle1>;
	static_assert(Traits0_t::isValid);
	static_assert(Traits1_t::isValid);
	GenVec<Element_t, opeDim> result;
	if constexpr (opeDim >= 1) { result.elements[0] = ldexp(rParam0.elements[Traits0_t::i0], rParam1.elements[Traits1_t::i0]); }
	if constexpr (opeDim >= 2) { result.elements[1] = ldexp(rParam0.elements[Traits0_t::i1], rParam1.elements[Traits1_t::i1]); }
	if constexpr (opeDim >= 3) { result.elements[2] = ldexp(rParam0.elements[Traits0_t::i2], rParam1.elements[Traits1_t::i2]); }
	if constexpr (opeDim >= 4) { result.elements[3] = ldexp(rParam0.elements[Traits0_t::i3], rParam1.elements[Traits1_t::i3]); }
	return result;
}


/*
	@@
	packUnorm2x16
	packSnorm2x16
	packUnorm4x8
	packSnorm4x8
	unpackUnorm2x16
	unpackSnorm2x16
	unpackUnorm4x8
	unpackSnorm4x8
	packHalf2x16
	unpackHalf2x16
	packDouble2x32
	unpackDouble2x32
	が未定義
*/


/* dot */
template<typename Element_t, int opeDim, int memDim0, int memDim1, int swizzle0, int swizzle1, int recursiveCount0, int recursiveCount1>
static inline Element_t dot(
	const Vec<Element_t, Traits<opeDim, memDim0, swizzle0>, recursiveCount0> &rParam0,
	const Vec<Element_t, Traits<opeDim, memDim1, swizzle1>, recursiveCount1> &rParam1
){
	using Traits0_t = Traits<opeDim, memDim0, swizzle0>;
	using Traits1_t = Traits<opeDim, memDim1, swizzle1>;
	static_assert(Traits0_t::isValid);
	static_assert(Traits1_t::isValid);
	Element_t tmp = 0;
	if constexpr (Traits0_t::opeDim >= 1) { tmp += rParam0.elements[Traits0_t::i0] * rParam1.elements[Traits1_t::i0]; }
	if constexpr (Traits0_t::opeDim >= 2) { tmp += rParam0.elements[Traits0_t::i1] * rParam1.elements[Traits1_t::i1]; }
	if constexpr (Traits0_t::opeDim >= 3) { tmp += rParam0.elements[Traits0_t::i2] * rParam1.elements[Traits1_t::i2]; }
	if constexpr (Traits0_t::opeDim >= 4) { tmp += rParam0.elements[Traits0_t::i3] * rParam1.elements[Traits1_t::i3]; }
	return tmp;
}
static inline float dot(float x, float y){
	return x * y;
}
static inline double dot(double x, double y){
	return x * y;
}


/* cross */
template<typename Element_t, int memDim0, int memDim1, int swizzle0, int swizzle1, int recursiveCount0, int recursiveCount1>
static inline GenVec<Element_t, 3> cross(
	const Vec<Element_t, Traits<3, memDim0, swizzle0>, recursiveCount0> &rParam0,
	const Vec<Element_t, Traits<3, memDim1, swizzle1>, recursiveCount1> &rParam1
){
	using Traits0_t = Traits<3, memDim0, swizzle0>;
	using Traits1_t = Traits<3, memDim1, swizzle1>;
	static_assert(Traits0_t::isValid);
	static_assert(Traits1_t::isValid);
	return
		GenVec<Element_t, 3>(
			rParam0.elements[Traits0_t::i1] * rParam1.elements[Traits1_t::i2] - rParam0.elements[Traits0_t::i2] * rParam1.elements[Traits1_t::i1],
			rParam0.elements[Traits0_t::i2] * rParam1.elements[Traits1_t::i0] - rParam0.elements[Traits0_t::i0] * rParam1.elements[Traits1_t::i2],
			rParam0.elements[Traits0_t::i0] * rParam1.elements[Traits1_t::i1] - rParam0.elements[Traits0_t::i1] * rParam1.elements[Traits1_t::i0]
		)
	;
}


/* length */
template<typename Element0_t, typename Traits0_t, int recursiveCount0>
static inline Element0_t length(
	const Vec<Element0_t, Traits0_t, recursiveCount0> &rVec
){
	static_assert(Traits0_t::isValid);
	return std::sqrt(dot(rVec, rVec));
}
static inline float length(float x){
	return x;
}
static inline double length(double x){
	return x;
}


/* distance */
template<typename Element_t, int opeDim, int memDim0, int memDim1, int swizzle0, int swizzle1, int recursiveCount0, int recursiveCount1>
static inline Element_t distance(
	const Vec<Element_t, Traits<opeDim, memDim0, swizzle0>, recursiveCount0> &rVec0,
	const Vec<Element_t, Traits<opeDim, memDim1, swizzle1>, recursiveCount1> &rVec1
){
	using Traits0_t = Traits<opeDim, memDim0, swizzle0>;
	using Traits1_t = Traits<opeDim, memDim1, swizzle1>;
	static_assert(Traits0_t::isValid);
	static_assert(Traits1_t::isValid);
	return length(rVec1 - rVec0);
}
static inline float distance(float x, float y){
	return std::abs(y - x);
}
static inline double distance(double x, double y){
	return std::abs(y - x);
}


/* normalize */
template<typename Element0_t, typename Traits0_t, int recursiveCount0>
static inline GenVec<Element0_t, Traits0_t::opeDim> normalize(
	const Vec<Element0_t, Traits0_t, recursiveCount0> &rVec
){
	static_assert(Traits0_t::isValid);
	Element0_t length2 = dot(rVec, rVec);
	return rVec * invertsqrt(length2);
}
static inline float normalize(float x){
	(void)x;
	return 1;
}
static inline double normalize(double x){
	(void)x;
	return 1;
}


/* reflect */
template<typename Element_t, int opeDim, int memDim0, int memDim1, int swizzle0, int swizzle1, int recursiveCount0, int recursiveCount1>
static inline GenVec<Element_t, opeDim> reflect(
	const Vec<Element_t, Traits<opeDim, memDim0, swizzle0>, recursiveCount0> &rVec0,
	const Vec<Element_t, Traits<opeDim, memDim1, swizzle1>, recursiveCount1> &rVec1
){
	using Traits0_t = Traits<opeDim, memDim0, swizzle0>;
	using Traits1_t = Traits<opeDim, memDim1, swizzle1>;
	static_assert(Traits0_t::isValid);
	static_assert(Traits1_t::isValid);
	return rVec0 - Element_t(2) * dot(rVec1, rVec0);
}
static inline float reflect(float param0, float param1){
	return param0 - 2.0f * dot(param1, param0);
}
static inline double reflect(double param0, double param1){
	return param0 - 2.0 * dot(param1, param0);
}


/* refract */
template<typename Element_t, int opeDim, int memDim0, int memDim1, int swizzle0, int swizzle1, int recursiveCount0, int recursiveCount1>
static inline GenVec<Element_t, opeDim> refract(
	const Vec<Element_t, Traits<opeDim, memDim0, swizzle0>, recursiveCount0> &rVec0,
	const Vec<Element_t, Traits<opeDim, memDim1, swizzle1>, recursiveCount1> &rVec1,
	Element_t eta
){
	using Traits0_t = Traits<opeDim, memDim0, swizzle0>;
	using Traits1_t = Traits<opeDim, memDim1, swizzle1>;
	static_assert(Traits0_t::isValid);
	static_assert(Traits1_t::isValid);

	Element_t dotNI = dot(rVec1, rVec0);
	Element_t k = 1 - eta * eta * (1 - dotNI * dotNI);
	if (k < 0) {
		return GenVec<Element_t, opeDim>(Element_t(0));
	} else {
		return eta * rVec0 - (eta * dotNI + std::sqrt(k)) * rVec1;
	}
}
static inline float refract(float param0, float param1, float eta){
	float dotNI = dot(param1, param0);
	float k = 1 - eta * eta * (1 - dotNI * dotNI);
	if (k < 0) {
		return 0;
	} else {
		return eta * param0 - (eta * dotNI + std::sqrt(k)) * param1;
	}
}
static inline double refract(double param0, double param1, double eta){
	double dotNI = dot(param1, param0);
	double k = 1 - eta * eta * (1 - dotNI * dotNI);
	if (k < 0) {
		return 0;
	} else {
		return eta * param0 - (eta * dotNI + std::sqrt(k)) * param1;
	}
}

/*
	@@
	faceforward

	8.8. Integer Functions
		uaddCarry
		usubBorrow
		umulExtended
		imulExtended
		bitfieldExtract
		bitfieldExtract
		bitfieldInsert
		bitfieldInsert
		bitfieldReverse
		bitfieldReverse
		bitCount
		findLSB
		findLSB
	が未定義
*/



/*=============================================================================
▼	行列用の関数定義
-----------------------------------------------------------------------------*/
/* matrixCompMult */
template<typename Element_t, int opeDimC, int memDimC0, int memDimC1, int opeDimR, int memDimR0, int memDimR1, int recursiveCount0, int recursiveCount1>
static inline GenMat<Element_t, opeDimC, opeDimR> matrixCompMult(
	const Mat<Element_t, opeDimC, memDimC0, opeDimR, memDimR0, recursiveCount0> &rParam0,
	const Mat<Element_t, opeDimC, memDimC1, opeDimR, memDimR1, recursiveCount1> &rParam1
){
	GenMat<Element_t, opeDimC, opeDimR> result;
	if constexpr (opeDimC >= 1) { result.columns[0] = rParam0.columns[0] * rParam1.columns[0]; }
	if constexpr (opeDimC >= 2) { result.columns[1] = rParam0.columns[1] * rParam1.columns[1]; }
	if constexpr (opeDimC >= 3) { result.columns[2] = rParam0.columns[2] * rParam1.columns[2]; }
	if constexpr (opeDimC >= 4) { result.columns[3] = rParam0.columns[3] * rParam1.columns[3]; }
	return result;
}

/* outerProduct */
template<typename Element_t, int opeDim0, int memDim0, int swizzle0, int opeDim1, int memDim1, int swizzle1, int recursiveCount0, int recursiveCount1>
static inline GenMat<Element_t, opeDim1, opeDim0> outerProduct(
	const Vec<Element_t, Traits<opeDim0, memDim0, swizzle0>, recursiveCount0> &rVec0,
	const Vec<Element_t, Traits<opeDim1, memDim1, swizzle1>, recursiveCount1> &rVec1
){
	/*
		       ■
		matC = ■ opeDim0
		       □
		       □


		       opeDim1

		matR = ■■■□

		                              opeDim1

		              ■              ■■■□
		matC * matR = ■ * ■■■□ = ■■■□ opeDim0
		              □              □□□□
		              □              □□□□
	*/
	using Traits1_t = Traits<opeDim1, memDim1, swizzle1>;
	GenMat<Element_t, 1, opeDim0> matC(rVec0);
	GenMat<Element_t, opeDim1, 1> matR;
	if constexpr (opeDim1 >= 1) { matR.columns[0].elements[0] = rVec1.elements[Traits1_t::i0]; }
	if constexpr (opeDim1 >= 2) { matR.columns[1].elements[0] = rVec1.elements[Traits1_t::i1]; }
	if constexpr (opeDim1 >= 3) { matR.columns[2].elements[0] = rVec1.elements[Traits1_t::i2]; }
	if constexpr (opeDim1 >= 4) { matR.columns[3].elements[0] = rVec1.elements[Traits1_t::i3]; }
	return matC * matR;
}

/* transpose */
template<typename Element_t, int opeDimC, int memDimC, int opeDimR, int memDimR, int recursiveCount>
static inline GenMat<Element_t, opeDimR, opeDimC> transpose(
	const Mat<Element_t, opeDimC, memDimC, opeDimR, memDimR, recursiveCount> &rParam
){
	GenMat<Element_t, opeDimR, opeDimC> result;
	if constexpr (opeDimC >= 1) {
		if constexpr (opeDimR >= 1) { result.columns[0].elements[0] = rParam.columns[0].elements[0]; }
		if constexpr (opeDimR >= 2) { result.columns[1].elements[0] = rParam.columns[0].elements[1]; }
		if constexpr (opeDimR >= 3) { result.columns[2].elements[0] = rParam.columns[0].elements[2]; }
		if constexpr (opeDimR >= 4) { result.columns[3].elements[0] = rParam.columns[0].elements[3]; }
	}
	if constexpr (opeDimC >= 2) {
		if constexpr (opeDimR >= 1) { result.columns[0].elements[1] = rParam.columns[1].elements[0]; }
		if constexpr (opeDimR >= 2) { result.columns[1].elements[1] = rParam.columns[1].elements[1]; }
		if constexpr (opeDimR >= 3) { result.columns[2].elements[1] = rParam.columns[1].elements[2]; }
		if constexpr (opeDimR >= 4) { result.columns[3].elements[1] = rParam.columns[1].elements[3]; }
	}
	if constexpr (opeDimC >= 3) {
		if constexpr (opeDimR >= 1) { result.columns[0].elements[2] = rParam.columns[2].elements[0]; }
		if constexpr (opeDimR >= 2) { result.columns[1].elements[2] = rParam.columns[2].elements[1]; }
		if constexpr (opeDimR >= 3) { result.columns[2].elements[2] = rParam.columns[2].elements[2]; }
		if constexpr (opeDimR >= 4) { result.columns[3].elements[2] = rParam.columns[2].elements[3]; }
	}
	if constexpr (opeDimC >= 4) {
		if constexpr (opeDimR >= 1) { result.columns[0].elements[3] = rParam.columns[3].elements[0]; }
		if constexpr (opeDimR >= 2) { result.columns[1].elements[3] = rParam.columns[3].elements[1]; }
		if constexpr (opeDimR >= 3) { result.columns[2].elements[3] = rParam.columns[3].elements[2]; }
		if constexpr (opeDimR >= 4) { result.columns[3].elements[3] = rParam.columns[3].elements[3]; }
	}
	return result;
}

/* determinant */
template<typename Element_t, int memDimC, int memDimR, int recursiveCount>
static inline Element_t determinant(
	const Mat<Element_t, 2, memDimC, 2, memDimR, recursiveCount> &rParam
){
	return
		rParam.columns[0].elements[0] * rParam.columns[1].elements[1]
	-	rParam.columns[1].elements[0] * rParam.columns[0].elements[1];
}
template<typename Element_t, int memDimC, int memDimR, int recursiveCount>
static inline Element_t determinant(
	const Mat<Element_t, 3, memDimC, 3, memDimR, recursiveCount> &rParam
){
	return
		rParam.columns[0].elements[0]
	*	(	rParam.columns[1].elements[1] * rParam.columns[2].elements[2]
		-	rParam.columns[2].elements[1] * rParam.columns[1].elements[2]
		)
	-	rParam.columns[1].elements[0]
	*	(	rParam.columns[0].elements[1] * rParam.columns[2].elements[2]
		-	rParam.columns[2].elements[1] * rParam.columns[0].elements[2]
		)
	+	rParam.columns[2].elements[0]
	*	(	rParam.columns[0].elements[1] * rParam.columns[1].elements[2]
		-	rParam.columns[1].elements[1] * rParam.columns[0].elements[2]
		)
	;
}
template<typename Element_t, int memDimC, int memDimR, int recursiveCount>
static inline Element_t determinant(
	const Mat<Element_t, 4, memDimC, 4, memDimR, recursiveCount> &rParam
){
	Element_t tmp0	= rParam.columns[2].elements[2] * rParam.columns[3].elements[3]
					- rParam.columns[3].elements[2] * rParam.columns[2].elements[3];
	Element_t tmp1	= rParam.columns[2].elements[1] * rParam.columns[3].elements[3]
					- rParam.columns[3].elements[1] * rParam.columns[2].elements[3];
	Element_t tmp2	= rParam.columns[2].elements[1] * rParam.columns[3].elements[2]
					- rParam.columns[3].elements[1] * rParam.columns[2].elements[2];
	Element_t tmp3	= rParam.columns[2].elements[0] * rParam.columns[3].elements[3]
					- rParam.columns[3].elements[0] * rParam.columns[2].elements[3];
	Element_t tmp4	= rParam.columns[2].elements[0] * rParam.columns[3].elements[2]
					- rParam.columns[3].elements[0] * rParam.columns[2].elements[2];
	Element_t tmp5	= rParam.columns[2].elements[0] * rParam.columns[3].elements[1]
					- rParam.columns[3].elements[0] * rParam.columns[2].elements[1];

	Element_t tmp6	=   rParam.columns[1].elements[1] * tmp0 - rParam.columns[1].elements[2] * tmp1 + rParam.columns[1].elements[3] * tmp2;
	Element_t tmp7	= -(rParam.columns[1].elements[0] * tmp0 - rParam.columns[1].elements[2] * tmp3 + rParam.columns[1].elements[3] * tmp4);
	Element_t tmp8	=   rParam.columns[1].elements[0] * tmp1 - rParam.columns[1].elements[1] * tmp3 + rParam.columns[1].elements[3] * tmp5;
	Element_t tmp9	= -(rParam.columns[1].elements[0] * tmp2 - rParam.columns[1].elements[1] * tmp4 + rParam.columns[1].elements[2] * tmp5);

	return
		rParam.columns[0].elements[0] * tmp6 + rParam.columns[0].elements[1] * tmp7
	+	rParam.columns[0].elements[2] * tmp8 + rParam.columns[0].elements[3] * tmp9
	;
}

/* inverse */
template<typename Element_t, int memDimC, int memDimR, int recursiveCount>
static inline GenMat<Element_t, 2, 2> inverse(
	const Mat<Element_t, 2, memDimC, 2, memDimR, recursiveCount> &rParam
){
	Element_t invDet = Element_t(1) / determinant(rParam);
	GenMat<Element_t, 2, 2> result = {
		{
			 rParam.columns[1].elements[1] * invDet,
			-rParam.columns[0].elements[1] * invDet
		},{
			-rParam.columns[1].elements[0] * invDet,
			 rParam.columns[0].elements[0] * invDet
		}
	};
	return result;
}
template<typename Element_t, int memDimC, int memDimR, int recursiveCount>
static inline GenMat<Element_t, 3, 3> inverse(
	const Mat<Element_t, 3, memDimC, 3, memDimR, recursiveCount> &rParam
){
	Element_t invDet = Element_t(1) / determinant(rParam);
	GenMat<Element_t, 3, 3> result = {
		{
			 (rParam.columns[1].elements[1] * rParam.columns[2].elements[2] - rParam.columns[2].elements[1] * rParam.columns[1].elements[2]) * invDet,
			-(rParam.columns[0].elements[1] * rParam.columns[2].elements[2] - rParam.columns[2].elements[1] * rParam.columns[0].elements[2]) * invDet,
			 (rParam.columns[0].elements[1] * rParam.columns[1].elements[2] - rParam.columns[1].elements[1] * rParam.columns[0].elements[2]) * invDet
		},{
			-(rParam.columns[1].elements[0] * rParam.columns[2].elements[2] - rParam.columns[2].elements[0] * rParam.columns[1].elements[2]) * invDet,
			 (rParam.columns[0].elements[0] * rParam.columns[2].elements[2] - rParam.columns[2].elements[0] * rParam.columns[0].elements[2]) * invDet,
			-(rParam.columns[0].elements[0] * rParam.columns[1].elements[2] - rParam.columns[1].elements[0] * rParam.columns[0].elements[2]) * invDet
		},{
			 (rParam.columns[1].elements[0] * rParam.columns[2].elements[1] - rParam.columns[2].elements[0] * rParam.columns[1].elements[1]) * invDet,
			-(rParam.columns[0].elements[0] * rParam.columns[2].elements[1] - rParam.columns[2].elements[0] * rParam.columns[0].elements[1]) * invDet,
			 (rParam.columns[0].elements[0] * rParam.columns[1].elements[1] - rParam.columns[1].elements[0] * rParam.columns[0].elements[1]) * invDet
		}
	};
	return result;
}
template<typename Element_t, int memDimC, int memDimR, int recursiveCount>
static inline GenMat<Element_t, 4, 4> inverse(
	const Mat<Element_t, 4, memDimC, 4, memDimR, recursiveCount> &m
){
/* @@ 要サニタイズ */
	typedef Element_t T;
	T Coef00 = m.columns[2].elements[2] * m.columns[3].elements[3] - m.columns[3].elements[2] * m.columns[2].elements[3];
	T Coef02 = m.columns[1].elements[2] * m.columns[3].elements[3] - m.columns[3].elements[2] * m.columns[1].elements[3];
	T Coef03 = m.columns[1].elements[2] * m.columns[2].elements[3] - m.columns[2].elements[2] * m.columns[1].elements[3];

	T Coef04 = m.columns[2].elements[1] * m.columns[3].elements[3] - m.columns[3].elements[1] * m.columns[2].elements[3];
	T Coef06 = m.columns[1].elements[1] * m.columns[3].elements[3] - m.columns[3].elements[1] * m.columns[1].elements[3];
	T Coef07 = m.columns[1].elements[1] * m.columns[2].elements[3] - m.columns[2].elements[1] * m.columns[1].elements[3];

	T Coef08 = m.columns[2].elements[1] * m.columns[3].elements[2] - m.columns[3].elements[1] * m.columns[2].elements[2];
	T Coef10 = m.columns[1].elements[1] * m.columns[3].elements[2] - m.columns[3].elements[1] * m.columns[1].elements[2];
	T Coef11 = m.columns[1].elements[1] * m.columns[2].elements[2] - m.columns[2].elements[1] * m.columns[1].elements[2];

	T Coef12 = m.columns[2].elements[0] * m.columns[3].elements[3] - m.columns[3].elements[0] * m.columns[2].elements[3];
	T Coef14 = m.columns[1].elements[0] * m.columns[3].elements[3] - m.columns[3].elements[0] * m.columns[1].elements[3];
	T Coef15 = m.columns[1].elements[0] * m.columns[2].elements[3] - m.columns[2].elements[0] * m.columns[1].elements[3];

	T Coef16 = m.columns[2].elements[0] * m.columns[3].elements[2] - m.columns[3].elements[0] * m.columns[2].elements[2];
	T Coef18 = m.columns[1].elements[0] * m.columns[3].elements[2] - m.columns[3].elements[0] * m.columns[1].elements[2];
	T Coef19 = m.columns[1].elements[0] * m.columns[2].elements[2] - m.columns[2].elements[0] * m.columns[1].elements[2];

	T Coef20 = m.columns[2].elements[0] * m.columns[3].elements[1] - m.columns[3].elements[0] * m.columns[2].elements[1];
	T Coef22 = m.columns[1].elements[0] * m.columns[3].elements[1] - m.columns[3].elements[0] * m.columns[1].elements[1];
	T Coef23 = m.columns[1].elements[0] * m.columns[2].elements[1] - m.columns[2].elements[0] * m.columns[1].elements[1];

	GenVec<T, 4> Fac0(Coef00, Coef00, Coef02, Coef03);
	GenVec<T, 4> Fac1(Coef04, Coef04, Coef06, Coef07);
	GenVec<T, 4> Fac2(Coef08, Coef08, Coef10, Coef11);
	GenVec<T, 4> Fac3(Coef12, Coef12, Coef14, Coef15);
	GenVec<T, 4> Fac4(Coef16, Coef16, Coef18, Coef19);
	GenVec<T, 4> Fac5(Coef20, Coef20, Coef22, Coef23);

	GenVec<T, 4> Vec0(m.columns[1].elements[0], m.columns[0].elements[0], m.columns[0].elements[0], m.columns[0].elements[0]);
	GenVec<T, 4> Vec1(m.columns[1].elements[1], m.columns[0].elements[1], m.columns[0].elements[1], m.columns[0].elements[1]);
	GenVec<T, 4> Vec2(m.columns[1].elements[2], m.columns[0].elements[2], m.columns[0].elements[2], m.columns[0].elements[2]);
	GenVec<T, 4> Vec3(m.columns[1].elements[3], m.columns[0].elements[3], m.columns[0].elements[3], m.columns[0].elements[3]);

	GenVec<T, 4> Inv0(Vec1 * Fac0 - Vec2 * Fac1 + Vec3 * Fac2);
	GenVec<T, 4> Inv1(Vec0 * Fac0 - Vec2 * Fac3 + Vec3 * Fac4);
	GenVec<T, 4> Inv2(Vec0 * Fac1 - Vec1 * Fac3 + Vec3 * Fac5);
	GenVec<T, 4> Inv3(Vec0 * Fac2 - Vec1 * Fac4 + Vec2 * Fac5);

	GenVec<T, 4> SignA(+1, -1, +1, -1);
	GenVec<T, 4> SignB(-1, +1, -1, +1);
	GenMat<T, 4, 4> Inverse(Inv0 * SignA, Inv1 * SignB, Inv2 * SignA, Inv3 * SignB);

	GenVec<T, 4> Row0(Inverse.columns[0].elements[0], Inverse.columns[1].elements[0], Inverse.columns[2].elements[0], Inverse.columns[3].elements[0]);

	GenVec<T, 4> Dot0(m.columns[0] * Row0);
	T Dot1 = (Dot0.x + Dot0.y) + (Dot0.z + Dot0.w);

	T OneOverDeterminant = static_cast<T>(1) / Dot1;

	return Inverse * OneOverDeterminant;
}


/*
	@@ この周辺、実装途中。
	https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.60.pdf
	をみながら関数を足していく。
*/


/*=============================================================================
▼	型短縮名
-----------------------------------------------------------------------------*/
using vec4 = GenVec<float, 4>;
using vec3 = GenVec<float, 3>;
using vec2 = GenVec<float, 2>;
using vec1 = GenVec<float, 1>;

using dvec4 = GenVec<double, 4>;
using dvec3 = GenVec<double, 3>;
using dvec2 = GenVec<double, 2>;
using dvec1 = GenVec<double, 1>;

using ivec4 = GenIvec<int, 4>;
using ivec3 = GenIvec<int, 3>;
using ivec2 = GenIvec<int, 2>;
using ivec1 = GenIvec<int, 1>;

using uvec4 = GenIvec<unsigned int, 4>;
using uvec3 = GenIvec<unsigned int, 3>;
using uvec2 = GenIvec<unsigned int, 2>;
using uvec1 = GenIvec<unsigned int, 1>;

using bvec4 = GenBvec<bool, 4>;
using bvec3 = GenBvec<bool, 3>;
using bvec2 = GenBvec<bool, 2>;
using bvec1 = GenBvec<bool, 1>;


using mat4x4 = GenMat<float, 4, 4>;
using mat4x3 = GenMat<float, 4, 3>;
using mat4x2 = GenMat<float, 4, 2>;
using mat4x1 = GenMat<float, 4, 1>;

using mat3x4 = GenMat<float, 3, 4>;
using mat3x3 = GenMat<float, 3, 3>;
using mat3x2 = GenMat<float, 3, 2>;
using mat3x1 = GenMat<float, 3, 1>;

using mat2x4 = GenMat<float, 2, 4>;
using mat2x3 = GenMat<float, 2, 3>;
using mat2x2 = GenMat<float, 2, 2>;
using mat2x1 = GenMat<float, 2, 1>;

using mat1x4 = GenMat<float, 1, 4>;
using mat1x3 = GenMat<float, 1, 3>;
using mat1x2 = GenMat<float, 1, 2>;
using mat1x1 = GenMat<float, 1, 1>;

using mat4 = GenMat<float, 4, 4>;
using mat3 = GenMat<float, 3, 3>;
using mat2 = GenMat<float, 2, 2>;
using mat1 = GenMat<float, 1, 1>;


using dmat4x4 = GenMat<double, 4, 4>;
using dmat4x3 = GenMat<double, 4, 3>;
using dmat4x2 = GenMat<double, 4, 2>;
using dmat4x1 = GenMat<double, 4, 1>;

using dmat3x4 = GenMat<double, 3, 4>;
using dmat3x3 = GenMat<double, 3, 3>;
using dmat3x2 = GenMat<double, 3, 2>;
using dmat3x1 = GenMat<double, 3, 1>;

using dmat2x4 = GenMat<double, 2, 4>;
using dmat2x3 = GenMat<double, 2, 3>;
using dmat2x2 = GenMat<double, 2, 2>;
using dmat2x1 = GenMat<double, 2, 1>;

using dmat1x4 = GenMat<double, 1, 4>;
using dmat1x3 = GenMat<double, 1, 3>;
using dmat1x2 = GenMat<double, 1, 2>;
using dmat1x1 = GenMat<double, 1, 1>;

using dmat4 = GenMat<double, 4, 4>;
using dmat3 = GenMat<double, 3, 3>;
using dmat2 = GenMat<double, 2, 2>;
using dmat1 = GenMat<double, 1, 1>;


};	/* namespace */


#endif		/* インクルードガード終端 */
